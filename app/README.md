# app/ — standalone prediction & inverse-design engine

`formulator_engine.py` is the sanitized core engine extracted from the deployed
web application. It contains only the reproducible science; the Streamlit user
interface, the authentication gate, the in-app "Nano Assistant" chatbot, and all
API keys have been left out (they live in the separate private application repo).

It is self-contained and runnable on its own:

```
python formulator_engine.py      # runs a short NSGA-II + SOP smoke test
```

## What it does

- loads the serialized production XGBoost models from `models/`,
- resolves a query drug to RDKit molecular descriptors,
- applies a k-nearest-neighbour applicability-domain gate (graded confidence),
- runs NSGA-II inverse design (Optuna `NSGAIISampler`, fixed seed) to return a
  Pareto front of candidate formulations (predicted size / EE / zeta potential),
- retrieves the nearest real published formulations to assemble an evidence-based
  Standard Operating Procedure, each traceable to its source DOI.

## Contents (self-contained)

```
formulator_engine.py
data/formulation_corpus.csv    # curated corpus used by the AD gate and SOP retriever
data/api_smiles.xlsx           # API name -> SMILES mapping
models/xgb_particle_size.joblib
models/xgb_encapsulation_efficiency.joblib
models/xgb_zeta_potential.joblib
```

The full, running application (UI + auth + assistant) is deployed at
https://cs-alg-smartformulator.streamlit.app
