"""
formulator_engine.py — ML + optimisation core for CS-ALG SmartFormulator (v4).

Decoupled from the Streamlit UI so it can be unit-tested standalone.

Pipeline:
  drug (name/SMILES) -> 12 RDKit descriptors
  -> k-NN Applicability-Domain gate (graded confidence)
  -> NSGA-II inverse design over composition + excipient/method choices
  -> Pareto-optimal formulations (predicted Size / EE / Zeta)
  -> k-NN retrieval of the nearest REAL published formulations (SOP + DOIs)

The three deployed models are XGBoost regressors (one per CQA) trained on the
27-feature BASE schema. Missing features are not used (BASE has none).
"""
from __future__ import annotations
import os, glob, functools
import numpy as np
import pandas as pd
import joblib
import requests

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "data", "formulation_corpus.csv")
SMILES_PATH = os.path.join(BASE_DIR, "data", "api_smiles.xlsx")
MODEL_PATHS = {
    "Size": os.path.join(BASE_DIR, "models", "xgb_particle_size.joblib"),
    "EE":   os.path.join(BASE_DIR, "models", "xgb_encapsulation_efficiency.joblib"),
    "Zeta": os.path.join(BASE_DIR, "models", "xgb_zeta_potential.joblib"),
}

# ============================================================
# FEATURE SCHEMA  (exact order the models expect)
# ============================================================
DESCRIPTORS = [
    "Effective SLogP", "Effective TPSA", "Effective AMW", "Effective NumRotBonds",
    "Effective NumHBD", "Effective NumHBA", "Effective NumAromaticRings",
    "Effective NumSaturatedRings", "Effective NumAliphaticRings", "Effective MolMR",
    "Effective FractionCSP3", "Effective NumHeteroatoms",
]
BASE_FEATURES = [
    "Chitosan Mass Fraction (%)", "Alginate Mass Fraction (%)", "Surfactant Mass Fraction (%)",
    "Crosslinker Mass Fraction (%)", "Total Solid Concentration (mg/mL)",
] + DESCRIPTORS + [
    "Total API Mass Fraction (%)", "Emulsion_Solvent_Evaporation", "Contains_Surfactant",
    "Surfactant_HLB", "Surfactant_MW", "Crosslinker_Cationic_Valency", "Crosslinker_Anic_Valency",
    "Crosslinker_Cationic_Charge_Density", "Crosslinker_Anic_Charge_Density", "Contains_Crosslinker",
]
TARGETS = {"Size": "Particle Size (nm)", "Zeta": "Zeta Potential (mV)", "EE": "Encapsulation Efficiency (%)"}

# categorical -> fixed encoded feature values (learned from the data)
CROSSLINKERS = {
    "CaCl2": dict(Crosslinker_Cationic_Valency=2, Crosslinker_Anic_Valency=0,
                  Crosslinker_Cationic_Charge_Density=0.018021, Crosslinker_Anic_Charge_Density=0.0,
                  Contains_Crosslinker=1),
    "TPP":   dict(Crosslinker_Cationic_Valency=0, Crosslinker_Anic_Valency=-5,
                  Crosslinker_Cationic_Charge_Density=0.0, Crosslinker_Anic_Charge_Density=-0.013592,
                  Contains_Crosslinker=1),
    "None":  dict(Crosslinker_Cationic_Valency=0, Crosslinker_Anic_Valency=0,
                  Crosslinker_Cationic_Charge_Density=0.0, Crosslinker_Anic_Charge_Density=0.0,
                  Contains_Crosslinker=0),
}
SURFACTANTS = {
    "Pluronic F127": dict(Contains_Surfactant=1, Surfactant_HLB=22, Surfactant_MW=12600),
    "Tween 80":      dict(Contains_Surfactant=1, Surfactant_HLB=15, Surfactant_MW=1310),
    "None":          dict(Contains_Surfactant=0, Surfactant_HLB=0, Surfactant_MW=0),
}
METHODS = {"Ionic gelation / PEC": 0, "Emulsification + ionic gelation": 1}

# ============================================================
# CACHED LOADERS
# ============================================================
@functools.lru_cache(maxsize=1)
def load_models():
    return {k: joblib.load(p) for k, p in MODEL_PATHS.items()}

@functools.lru_cache(maxsize=1)
def load_corpus():
    return pd.read_csv(CORPUS_PATH)

@functools.lru_cache(maxsize=1)
def load_training_smiles():
    """Return {api_name_lower: smiles} for the training drugs."""
    try:
        s = pd.read_excel(SMILES_PATH)
        name_col = next(c for c in s.columns if "name" in c.lower() or "api" in c.lower())
        smi_col = next(c for c in s.columns if "smiles" in c.lower())
        return {str(r[name_col]).strip().lower(): str(r[smi_col]).strip()
                for _, r in s.iterrows() if pd.notna(r[smi_col])}
    except Exception:
        return {}

def bounds_from_corpus():
    """Decision-variable bounds pulled from the observed data envelope."""
    df = load_corpus()
    b = {c: (float(df[c].min()), float(df[c].max())) for c in
         ["Chitosan Mass Fraction (%)", "Alginate Mass Fraction (%)", "Surfactant Mass Fraction (%)",
          "Crosslinker Mass Fraction (%)", "Total API Mass Fraction (%)", "Total Solid Concentration (mg/mL)"]}
    return b

# ============================================================
# SMILES + DESCRIPTORS
# ============================================================
def resolve_smiles(query: str, by_smiles: bool = False):
    """Resolve a compound name to canonical SMILES (PubChem -> NCI CACTUS).
    If by_smiles=True, just validate/return the given SMILES."""
    q = (query or "").strip()
    if not q:
        return None
    if by_smiles:
        return q if Chem.MolFromSmiles(q) else None
    # training list first (offline, exact)
    tr = load_training_smiles().get(q.lower())
    if tr:
        return tr
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property/CanonicalSMILES/JSON",
            timeout=6)
        if r.status_code == 200:
            return r.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception:
        pass
    try:
        r = requests.get(f"https://cactus.nci.nih.gov/chemical/structure/{q}/smiles", timeout=6)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
    except Exception:
        pass
    return None

def compute_descriptors(smiles: str):
    """12 RDKit 2-D descriptors, matching the training pipeline exactly."""
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return None
    return {
        "Effective SLogP": Descriptors.MolLogP(mol),
        "Effective TPSA": Descriptors.TPSA(mol),
        "Effective AMW": Descriptors.MolWt(mol),
        "Effective NumRotBonds": Descriptors.NumRotatableBonds(mol),
        "Effective NumHBD": Descriptors.NumHDonors(mol),
        "Effective NumHBA": Descriptors.NumHAcceptors(mol),
        "Effective NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Effective NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "Effective NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Effective MolMR": Descriptors.MolMR(mol),
        "Effective FractionCSP3": Descriptors.FractionCSP3(mol),
        "Effective NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
    }

# ============================================================
# APPLICABILITY DOMAIN  (Enalos k-NN on the 12 descriptors)
# ============================================================
class ADGate:
    def __init__(self, k: int = 5):
        self.k = k
    def fit(self):
        df = load_corpus()
        X = df[DESCRIPTORS].drop_duplicates().dropna()
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean").fit(Xs)
        d, _ = self.nn.kneighbors(Xs)
        md = d[:, 1:].mean(axis=1)
        self.threshold = float(md.mean() + 2 * md.std())
        self.n_ref = len(X)
        return self
    def check(self, descriptors: dict):
        xs = self.scaler.transform(pd.DataFrame([descriptors])[DESCRIPTORS])
        d, _ = self.nn.kneighbors(xs, n_neighbors=self.k - 1)
        dist = float(d.mean())
        ratio = dist / self.threshold
        if ratio <= 0.7:   level, msg = "high", "High confidence — well inside the training domain."
        elif ratio <= 1.0: level, msg = "moderate", "Moderate — near the edge of the training domain."
        else:              level, msg = "outside", "Outside the domain — predictions are NOT reliable."
        return dict(distance=round(dist, 3), threshold=round(self.threshold, 3),
                    ratio=round(ratio, 2), level=level, in_domain=ratio <= 1.0, message=msg)

@functools.lru_cache(maxsize=1)
def get_ad_gate():
    return ADGate().fit()

# ============================================================
# FEATURE ROW ASSEMBLY + PREDICTION
# ============================================================
def assemble_row(descriptors, mass_fracs, total_solid_conc, crosslinker, surfactant, method):
    """mass_fracs: dict with the 5 fractions (already summing to ~100)."""
    row = dict(descriptors)
    row["Chitosan Mass Fraction (%)"] = mass_fracs["Chitosan"]
    row["Alginate Mass Fraction (%)"] = mass_fracs["Alginate"]
    row["Surfactant Mass Fraction (%)"] = mass_fracs["Surfactant"]
    row["Crosslinker Mass Fraction (%)"] = mass_fracs["Crosslinker"]
    row["Total API Mass Fraction (%)"] = mass_fracs["API"]
    row["Total Solid Concentration (mg/mL)"] = total_solid_conc
    row["Emulsion_Solvent_Evaporation"] = METHODS[method]
    row.update(SURFACTANTS[surfactant])
    row.update(CROSSLINKERS[crosslinker])
    return row

def predict_all(models, row):
    X = pd.DataFrame([row])[BASE_FEATURES]
    return {k: float(m.predict(X)[0]) for k, m in models.items()}

# ============================================================
# NSGA-II INVERSE DESIGN
# ============================================================
def run_inverse_design(descriptors, n_trials=400, seed=42,
                       allowed_crosslinkers=("CaCl2", "TPP", "None"),
                       allowed_surfactants=("Pluronic F127", "Tween 80", "None"),
                       allowed_methods=("Ionic gelation / PEC", "Emulsification + ionic gelation"),
                       progress=None):
    """Multi-objective inverse design. Returns a DataFrame of Pareto-optimal formulations.
    `progress` (optional): a plain dict mutated in place with a 'done' counter after every
    trial so a background-thread runner can poll live progress."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    models = load_models()
    b = bounds_from_corpus()
    CSmax = b["Chitosan Mass Fraction (%)"][1]
    ALGmax = b["Alginate Mass Fraction (%)"][1]
    SURFmax = b["Surfactant Mass Fraction (%)"][1]
    CXmax = b["Crosslinker Mass Fraction (%)"][1]
    APImin, APImax = b["Total API Mass Fraction (%)"]
    TSCmin, TSCmax = b["Total Solid Concentration (mg/mL)"]

    def objective(trial):
        xl = trial.suggest_categorical("crosslinker", list(allowed_crosslinkers))
        sf = trial.suggest_categorical("surfactant", list(allowed_surfactants))
        mt = trial.suggest_categorical("method", list(allowed_methods))
        w_cs = trial.suggest_float("w_cs", 0.0, CSmax)
        w_alg = trial.suggest_float("w_alg", 0.0, ALGmax)
        w_sf = trial.suggest_float("w_sf", 0.0, SURFmax)
        w_cx = trial.suggest_float("w_cx", 0.0, CXmax)
        w_api = trial.suggest_float("w_api", max(APImin, 0.05), APImax)
        if sf == "None": w_sf = 0.0
        if xl == "None": w_cx = 0.0
        tot = w_cs + w_alg + w_sf + w_cx + w_api
        if tot <= 0:
            return -1e9, 1e9, -1e9
        f = lambda w: w / tot * 100.0
        mass = dict(Chitosan=f(w_cs), Alginate=f(w_alg), Surfactant=f(w_sf),
                    Crosslinker=f(w_cx), API=f(w_api))
        tsc = trial.suggest_float("tsc", TSCmin, TSCmax)
        row = assemble_row(descriptors, mass, tsc, xl, sf, mt)
        p = predict_all(models, row)
        trial.set_user_attr("zeta_signed", p["Zeta"])
        trial.set_user_attr("mass", mass)
        trial.set_user_attr("tsc", tsc)
        trial.set_user_attr("choices", (xl, sf, mt))
        return p["EE"], p["Size"], abs(p["Zeta"])

    def _cb(study_, trial_):
        if progress is not None:
            progress["done"] = progress.get("done", 0) + 1

    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(directions=["maximize", "minimize", "maximize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False, callbacks=[_cb])

    rows = []
    for t in study.best_trials:
        xl, sf, mt = t.user_attrs["choices"]
        m = t.user_attrs["mass"]
        rows.append({
            "Chitosan Mass Fraction (%)": round(m["Chitosan"], 2),
            "Alginate Mass Fraction (%)": round(m["Alginate"], 2),
            "Surfactant Mass Fraction (%)": round(m["Surfactant"], 2),
            "Crosslinker Mass Fraction (%)": round(m["Crosslinker"], 2),
            "API Mass Fraction (%)": round(m["API"], 2),
            "Total Solid Conc (mg/mL)": round(t.user_attrs["tsc"], 2),
            "Crosslinker": xl, "Surfactant": sf, "Method": mt,
            "Pred_EE (%)": round(t.values[0], 1),
            "Pred_Size (nm)": round(t.values[1], 1),
            "Pred_Zeta (mV)": round(t.user_attrs["zeta_signed"], 1),
        })
    return pd.DataFrame(rows)

# ============================================================
# SOP RETRIEVAL  (nearest real published formulations)
# ============================================================
GRADE = {0: "Oligosaccharide", 1: "Low MW", 2: "Medium MW", 3: "High MW"}

def _decode_crosslinker(r):
    if r["Contains_Crosslinker"] == 0: return "None (polymer self-crosslinking)"
    if r["Crosslinker_Cationic_Valency"] == 2: return "CaCl2"
    if r["Crosslinker_Anic_Valency"] == -5: return "TPP / STPP"
    return "Other"

def _decode_surfactant(r):
    if r["Contains_Surfactant"] == 0: return "None"
    if r["Surfactant_HLB"] == 22: return "Pluronic F127"
    if r["Surfactant_HLB"] == 15: return "Tween 80"
    return f"HLB {r['Surfactant_HLB']}"

# process-parameter fields: (neighbour key, corpus column, display label, decimals)
_PROC_CONT = [
    ("cs_dda", "Chitosan_DDA_Pct", "Chitosan DDA (%)", 0),
    ("pH", "pH_Standardized", "pH", 1),
    ("rpm", "Stirring_Speed_RPM", "Stirring (rpm)", 0),
    ("temp", "Temperature_C", "Temperature (°C)", 0),
    ("sonication", "Sonication_Time_min", "Sonication (min)", 0),
]


class SOPRetriever:
    def fit(self):
        self.df = load_corpus().reset_index(drop=True)
        self.scaler = StandardScaler().fit(self.df[BASE_FEATURES])
        self.Xs = self.scaler.transform(self.df[BASE_FEATURES])
        self.nn = NearestNeighbors(metric="euclidean").fit(self.Xs)
        # fallback medians for fields no neighbour reports: grouped by the
        # emulsion/no-emulsion flag (the strongest process-context signal),
        # plus a global median as a last resort.
        self._global_med = {col: float(self.df[col].median()) for _, col, _, _ in _PROC_CONT}
        self._grp_med = {}
        for flag, sub in self.df.groupby("Emulsion_Solvent_Evaporation"):
            self._grp_med[int(flag)] = {col: (float(sub[col].median()) if sub[col].notna().any()
                                              else self._global_med[col])
                                        for _, col, _, _ in _PROC_CONT}
        return self

    def retrieve(self, formulation_row: dict, k: int = 8):
        """Nearest real published formulations (de-duplicated), for pooling."""
        x = pd.DataFrame([formulation_row])[BASE_FEATURES]
        d, idx = self.nn.kneighbors(self.scaler.transform(x), n_neighbors=min(60, len(self.df)))
        out, seen = [], set()
        for dist, i in zip(d[0], idx[0]):
            r = self.df.iloc[i]
            key = (r["API Name"], r.get("DOI"))
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(distance=round(float(dist), 3), api=r["API Name"], doi=r.get("DOI", ""),
                            paper=str(r.get("Paper Name", ""))[:90],
                            crosslinker=_decode_crosslinker(r), surfactant=_decode_surfactant(r),
                            method=("Emulsification + ionic gelation" if r["Emulsion_Solvent_Evaporation"] == 1 else "Ionic gelation / PEC"),
                            chitosan_grade=GRADE.get(r.get("Chitosan_MW_Grade_Encoded"), "n.r."),
                            cs_dda=r.get("Chitosan_DDA_Pct"), pH=r.get("pH_Standardized"),
                            rpm=r.get("Stirring_Speed_RPM"), temp=r.get("Temperature_C"),
                            sonication=r.get("Sonication_Time_min")))
            if len(out) >= k:
                break
        return out

    def build_sop(self, neighbours):
        """Pool EACH parameter independently across the neighbours, skipping
        blanks. Continuous fields report a weighted centre + observed range +
        how many of the neighbours actually reported it; if none did, fall back
        to the method-grouped median (flagged 'typical'). Categorical fields use
        a distance-weighted mode."""
        if not neighbours:
            return {}
        w = np.array([1.0 / (n["distance"] + 1e-3) for n in neighbours])

        def wmode(key):
            from collections import defaultdict
            acc = defaultdict(float)
            for n, ww in zip(neighbours, w):
                acc[n[key]] += ww
            return max(acc, key=acc.get)

        method_mode = wmode("method")
        flag = 1 if method_mode.startswith("Emulsi") else 0

        out = {"Method": {"value": method_mode, "detail": "", "source": "reported"},
               "Crosslinker": {"value": wmode("crosslinker"), "detail": "", "source": "reported"},
               "Surfactant": {"value": wmode("surfactant"), "detail": "", "source": "reported"},
               "Chitosan grade": {"value": wmode("chitosan_grade"), "detail": "", "source": "reported"}}

        for key, col, label, dec in _PROC_CONT:
            pairs = [(float(n[key]), ww) for n, ww in zip(neighbours, w) if pd.notna(n[key])]
            if pairs:
                vals = np.array([v for v, _ in pairs]); ws = np.array([x for _, x in pairs])
                centre = float(np.average(vals, weights=ws))
                lo, hi, n_rep = vals.min(), vals.max(), len(vals)
                fmt = (lambda v: f"{v:.{dec}f}")
                span = fmt(lo) if lo == hi else f"{fmt(lo)} to {fmt(hi)}"
                # clean value for the card; rich provenance kept only in `detail`
                out[label] = {"value": fmt(centre),
                              "detail": (f"typical range {span} across "
                                         f"{n_rep} similar stud{'y' if n_rep == 1 else 'ies'}"),
                              "source": "reported"}
            else:
                fb = self._grp_med.get(flag, self._global_med)[col]
                out[label] = {"value": f"{fb:.{dec}f}",
                              "detail": "estimated from similar formulations (not directly reported)",
                              "source": "fallback"}
        return out

@functools.lru_cache(maxsize=1)
def get_sop_retriever():
    return SOPRetriever().fit()

def formulation_to_feature_row(formulation: dict, descriptors: dict):
    """Turn a Pareto row (with human labels) back into the 27-feature vector for retrieval."""
    mass = dict(Chitosan=formulation["Chitosan Mass Fraction (%)"],
                Alginate=formulation["Alginate Mass Fraction (%)"],
                Surfactant=formulation["Surfactant Mass Fraction (%)"],
                Crosslinker=formulation["Crosslinker Mass Fraction (%)"],
                API=formulation["API Mass Fraction (%)"])
    return assemble_row(descriptors, mass, formulation["Total Solid Conc (mg/mL)"],
                        formulation["Crosslinker"], formulation["Surfactant"], formulation["Method"])


# ============================================================
# SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("Models:", list(load_models())); print("Corpus:", load_corpus().shape)
    smi = load_training_smiles(); print("Training SMILES loaded:", len(smi))
    # pick a known drug
    name = "Curcumin diglutaric acid"
    s = resolve_smiles(name)
    if not s:
        s = list(smi.values())[0]; name = "(first training drug)"
    desc = compute_descriptors(s)
    print(f"\nDrug: {name}\n descriptors AMW={desc['Effective AMW']:.1f} SLogP={desc['Effective SLogP']:.2f}")
    print(" AD:", get_ad_gate().check(desc))
    print("\nRunning NSGA-II (120 trials for the smoke test)...")
    pareto = run_inverse_design(desc, n_trials=120, seed=1)
    print(f" Pareto formulations: {len(pareto)}")
    print(pareto.head(5).to_string(index=False))
    if len(pareto):
        top = pareto.iloc[0].to_dict()
        row = formulation_to_feature_row(top, desc)
        nbrs = get_sop_retriever().retrieve(row, k=3)
        print("\nSOP nearest real formulations:")
        for n in nbrs:
            print(f"  d={n['distance']:.2f} {n['api'][:24]:24s} {n['crosslinker']:10s} pH {n['pH']} {n['rpm']}rpm DOI {n['doi']}")
        print(" Aggregated SOP:", get_sop_retriever().build_sop(nbrs))
