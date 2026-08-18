# CS-ALG SmartFormulator

Reproducibility repository for the manuscript:

> **An End-to-End Automated Pipeline for Inverse Design of Chitosan–Alginate
> Nanoparticle Drug-Delivery Systems: From Large Language Model-Based Literature
> Extraction to a Deployed Web Application.**
> Yasintha Lakshan Ranasinghe Ranepura Hewage, Monsin Sangsawat, Adisorn Sirichotejirakul,
> John Wilfred T. Malabanan, Yilin Song, Pornchai Rojsitthisak, Pranee Rojsitthisak.

This repository contains everything needed to reproduce the development pipeline
(literature extraction → curation → featurization → nested-CV model training →
feature attribution) and the core prediction/inverse-design logic behind the
deployed tool.

**Live web application:** https://cs-alg-smartformulator.streamlit.app

The full Streamlit web application (user interface, deployment configuration, and
the optional in-app conversational assistant) is maintained in a separate,
private repository. This repository publishes the reproducible science: the
development workflow, the trained models, the curated data, and the standalone
prediction/inverse-design engine.

## Repository layout

```
knime/    CSALG_KNIME_workflow.knwf  — the full development workflow (KNIME 5.8.3 LTS)
scripts/  the 21 Python scripts embedded in the workflow's Python Script nodes
app/      self-contained inverse-design engine: formulator_engine.py, plus its own
          data/ (corpus + SMILES) and models/ (the three production XGBoost models)
data/     provenance datasets: curated data, raw extraction outputs, API SMILES, extraction prompt
```

## Requirements

- KNIME Analytics Platform 5.8.3 LTS (to open `knime/CSALG_KNIME_workflow.knwf`)
- Python 3.12 with the packages in `requirements.txt`
- IBM Docling (for the PDF-to-Markdown stage)

Install the Python dependencies with:

```
pip install -r requirements.txt
```

The exact conda environment used to produce the reported results (a full package
listing with versions) is provided in `conda_environment_packages.md`.

**Reproducibility note.** All estimators are run single-threaded (`n_jobs = 1`)
with fixed seeds (`random_state = 42`, `TPESampler(seed = 42)`). This guarantees
bit-level reproducibility of the reported metrics; parallel execution
(`n_jobs = -1`) can perturb Random Forest predictions through the accumulation
order of floating-point operations across threads.

## Reproducing the pipeline

The KNIME workflow is the executable artifact; the scripts in `scripts/` are the
code run inside its Python Script nodes, provided separately so each stage can be
read without installing KNIME. The mapping from script to workflow stage to the
Methods section of the paper is below.

| Script | Stage | Methods |
|---|---|---|
| 1. Automated PDF-to-Markdown Conversion via Docling | PDF → structured Markdown | 2.3 |
| 2. Fault-Tolerant LLM JSON Parser & Relational Schema Mapper | Parse LLM JSON to a relational table | 2.5 |
| 3. API Standardization & Macromolecule Filter | Restrict to small-molecule drugs | 2.5 |
| 4. Formulation Feature Engineering & Scale-Invariant Imputation Pipeline | Composition feature engineering | 2.5–2.6 |
| 5. UNIT Curation | Unit standardization to mass fractions | 2.5 |
| 6. RDKit 2D Descriptor Generation & Weighted API Consolidation | Molecular descriptors; "effective" descriptors | 2.6 |
| 7. Method Categorization & Outlier Pruning (One-Hot Encoding) | Method encoding; per-target outlier removal | 2.5 |
| 8. Excipient Characterization (Surfactant & Crosslinker Physical Descriptors) | HLB/MW and charge encoding of excipients | 2.6.1 |
| 9. Polymer Characterization & NLP Extraction | Chitosan DDA / MW-grade extraction | 2.6, 2.7 |
| 10. Process Parameter Standardization & Curation | Process-parameter cleanup | 2.5, 2.7 |
| 11. Stratified K-Fold Splitter for Continuous Targets | Stratified 5-fold construction | 2.8 |
| 12. Random Forest — Optuna hyperparameter optimization | Inner-loop tuning (nested CV) | 2.8 |
| 13. Random Forest — Outer-Loop Training & Generalization Evaluation | Outer-loop nested-CV scoring | 2.8 |
| 14. Random Forest — Global Hyperparameter Optimization | Flat 5-fold tuning for production | 2.9 |
| 15. Random Forest — Model Training with Full Data Set | Final full-data model, serialized | 2.9 |
| 16. Random Forest — RFECV | Feature attribution | 2.10 |
| 17. XGBoost — Optuna hyperparameter optimization | Inner-loop tuning (nested CV) | 2.8 |
| 18. XGBoost — Outer-Loop Training & Generalization Evaluation | Outer-loop nested-CV scoring | 2.8 |
| 19. XGBoost — Global Hyperparameter Optimization | Flat 5-fold tuning for production | 2.9 |
| 20. XGBoost — Model Training with Full Data Set | Final full-data model, serialized | 2.9 |
| 21. XGBoost — RFECV | Feature attribution | 2.10 |

The full verbatim LLM extraction prompt (the schema-locked system prompt of
Section 2.4) is provided in `data/extraction_prompt.txt` and as Supplementary
Note S1 of the paper.

## Data

| File | Description |
|---|---|
| `data/Curated_dataset_full_402rows.csv` | The curated dataset (402 formulations, 23 APIs) |
| `data/Base_model_dataset_402rows.csv` | The 27 completely-reported base features used for modelling |
| `data/LLM_extracted_raw_715rows.csv` | Raw parsed extraction output before curation |
| `data/LLM_Responses_public.csv` | Per-article LLM responses (source text removed for copyright) |
| `data/API_SMILES.xlsx` | API name → SMILES mapping used for descriptor generation |
| `data/extraction_prompt.txt` | The schema-locked extraction prompt (Supplementary Note S1) |

## API key

The extraction stage calls a large language model through KNIME's
OpenAI-compatible nodes. **No API key is stored in this repository.** To re-run
the extraction, supply your own OpenAI-compatible API key in the workflow's
Credentials Configuration node at run time.

**Important:** in the Credentials Configuration node, tick **"Save password in
configuration (weakly encrypted)"** so the key is propagated to the downstream
LLM nodes at execution time. Without this option enabled the credential is not
passed through and the extraction nodes will fail to authenticate.

## What is intentionally not included

- The Streamlit user-interface and deployment configuration (kept in the private
  application repository; the live tool is linked above).
- The optional in-app conversational assistant ("Nano Assistant") and any
  associated model keys — it does not contribute to the study's analyses or
  results.
- Any API keys, tokens, or secrets.

## License

- Code (`knime/`, `scripts/`, `app/formulator_engine.py`): MIT License (see `LICENSE`).
- Data (`data/`, `app/data/`) and trained models (`app/models/`): Creative Commons
  Attribution 4.0 International (CC BY 4.0).

## Citation

If you use this repository, please cite the associated paper (see `CITATION.cff`).
