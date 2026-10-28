# Cybersecurity Intrusion Detection Using Machine Learning

End-to-end machine-learning project (**ECSCI24302 - Problem-Based Learning**) that
detects malicious network flows using the **UNSW-NB15** dataset from Kaggle/UCI.
It covers dataset verification, a data-quality audit, EDA, leaks-safe preprocessing,
a dummy baseline plus four ML models, hyper-parameter tuning, threshold selection, a
**single** untouched-test pass, model comparison, error analysis, interpretation,
robustness/leakage checks, an optional multi-class extension and a Streamlit dashboard.

## Quick start

```bash
# 1. Environment (Python 3.12 used / tested)
python -m venv .venv
.venv\Scripts\activate               # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Dataset (not redistributed with the repo)
python -m src.download_data          # downloads + verifies UNSW-NB15, ~830k combined rows

# 3. Full reproducible pipeline (EDA -> dedup -> tuning -> threshold -> single test pass)
python -m src.train                  # ~6.2 min on a 20-core machine; writes results/ + artifacts/

# 4. Interactive outputs
jupyter lab notebooks\\Cybersecurity_Intrusion_Detection.ipynb   # fully executed notebook
streamlit run app\\app.py                                        # 4-page dashboard
pytest                            # pipeline smoke tests
```

## Dataset

| Item | Value |
|---|---|
| Name | **UNSW-NB15** - official partitioned release (UNSW Canberra Cyber, 2015) |
| Access page | https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 |
| Download mirror used by `src/download_data.py` | https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15 (no credentials; verified byte/row-compatible) |
| Rows | Training partition 175,341 (deduplicated -> 107,740) &middot; Testing partition 82,332 (kept exactly as released) |
| Columns | 45 per file (49 numeric flow features + 3 categorical in the modeling schema) |
| Classes | Binary: benign vs malicious; attack families: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms |
| Accessed date | 2026-10-26 |
| License | Public academic release (verify the exact licence on the access page) |

## Leakage-control rules (enforced in code)

1. The official **training** partition is split before fitting data-dependent
   preprocessing; imputers/scalers/encoders are fitted on training data only.
2. The official **testing** partition is never cleaned, never used for tuning,
   model selection or threshold decisions and is evaluated exactly once.
3. `id`, `label`, `attack_cat` are excluded from model inputs - no target leakage.
4. All models share one `Pipeline` shape (features -> preprocess -> model).
5. Every number shown anywhere is quoted from `results/findings.md` or sibling
   CSV/JSON files - never estimated or invented.

## Repository layout

```
Cybersecurity_Intrusion_Detection/
├── app/                     Streamlit dashboard (app.py)
├── artifacts/               Fitted pipeline + metadata (joblib/json)
├── data/                    UNSW-NB15 CSVs (downloaded; not for redistribution)
├── docs/                    Architecture diagram + notes
├── notebooks/               Fully executed end-to-end notebook
├── proposal/                Project proposal (DOCX)
├── report/                  Final report (DOCX)
├── presentation/            Slide deck (PPTX)
├── results/                 findings.md, model_comparison.csv, figures/ ...
├── scripts/                 Builders (notebook/documents) + packaging
├── src/                     data_loading, preprocessing, eda, evaluate, train, predict
└── tests/                   pytest smoke tests
```

## Headline results (verify against `results/findings.md`)

- Selected model: **HistGradientBoosting (tuned)** - test accuracy 0.8712,
  precision 0.8154, recall 0.9903, F1 0.8944, ROC-AUC 0.9868, threshold 0.480.
- Honest finding: FPR is 27.47% (false alarms) vs FNR 0.97% (missed attacks) - the
  official train/test distribution shift is documented as a limitation.
- Optional 10-family classifier: accuracy 0.6814, macro-F1 0.5129 (Analysis and DoS
  recall are hard). Leakage sensitivity: 5.16% shared exact flows, F1 0.8944 -> 0.8902.

## Notes

- Scores are deterministic (`random_state=42`).
- Data is downloaded, not shipped - see `src/download_data.py`.
- See `Viva_Preparation.md` for an examiner-style Q&A review list.
