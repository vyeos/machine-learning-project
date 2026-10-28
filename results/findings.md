# Experiment findings - single source of truth
Generated automatically by `python -m src.train` on 2026-10-26 21:47.
Notebooks, the report and the slides must quote numbers exactly from this file
(or from the sibling CSV/JSON files) - never estimate or invent them.

## 1. Dataset (verified facts)
- Name: **UNSW-NB15 (official partitioned release)**; Kaggle listing: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
- Training partition raw rows: **175,341** (deduplicated -> **107,740**)
- Testing partition rows (untouched benchmark): **82,332**
- Columns per file: 45 (official release schema has 45)
- id range train [1, 175341] (identifier, excluded from modeling); test [1, 82332]
- Missing cells (raw files): train 0, test 0 -> **verified: no imputation needed**
- Non-finite numeric cells: train none, test none
- Input feature columns for modeling: **52** = 49 numeric + 3 categorical
- Excluded from modeling: id (identifier), label (target), attack_cat (target-derived annotation)
- Engineered features: **16** added by FeatureEngineer inside pipelines
- Attack classes present: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms
- Label/attack_cat consistency: 0 + 0 = 0 verified mismatches

## 2. Class balance (figures 1-2)
| Split | Benign (0) | Malicious (1) | Attack share | Max:Min ratio |
|---|---|---|---|---|
| Training | 56,000 | 119,341 | 68.06% | 2.13 |
| Testing | 37,000 | 45,332 | 55.06% | - |

## 3. Data-quality findings (figures 6-7)
- Exact duplicate rows in training (excluding id): **67,601** - removed, reported
- Exact duplicate rows in testing (excluding id): **26,387** - kept (deployment realism; testing never cleaned)
- Distinct training flow tuples that also appear in testing: **940** - quantified in the leakage sensitivity check
- Categorical cardinality (train): {'proto': {'train': 133, 'test': 131}, 'service': {'train': 13, 'test': 13}, 'state': {'train': 9, 'test': 7}}
- Unseen test categories: {'proto': [], 'service': [], 'state': ['ACC', 'CLO']}
- Flag inventory: is_ftp_login {'0': 172774, '1': 2545, '2': 6, '4': 16}; ct_ftp_cmd top counts {'0': 172774, '1': 2545, '2': 6, '4': 16}; is_sm_ips_ports {'0': 172579, '1': 2762}
- Negative numeric cell counts: none
- Protocol attack-share (top categories): tcp: 51.07%, udp: 78.00%, unas: 100.00%, arp: 0.00%, ospf: 97.53%, sctp: 100.00%

## 4. EDA highlights (figures 3-5)
Key feature medians (benign vs malicious):
| Feature | Median benign | Median malicious |
|---|---|---|
| dur | 0.0386 | 9e-06 |
| rate | 1.38e+03 | 1.11e+05 |
| sbytes | 1.47e+03 | 200 |
| dbytes | 1.11e+03 | 0 |
| sttl | 31 | 254 |
| dttl | 29 | 0 |
| sload | 4.31e+05 | 5.07e+07 |
| dload | 3.36e+05 | 0 |
| swin | 255 | 0 |
| dwin | 255 | 0 |
| tcprtt | 0.000645 | 0 |
| ct_dst_src_ltm | 2 | 5 |
| ct_srv_src | 4 | 5 |
| ct_state_ttl | 0 | 2 |

- Features most positively correlated with 'malicious': **sttl** (0.693), **ct_state_ttl** (0.578), **ct_dst_sport_ltm** (0.357), **rate** (0.338), **ct_src_dport_ltm** (0.306), **ct_dst_src_ltm** (0.304), **ct_src_ltm** (0.238), **ct_dst_ltm** (0.230)
- Features most negatively correlated with 'malicious': **dload** (-0.394), **dmean** (-0.342), **swin** (-0.334), **dwin** (-0.320), **stcpb** (-0.255), **dtcpb** (-0.250)
- Differences are descriptive, not proof of predictiveness.

## 5. Cross-validation on the training partition (5-fold stratified)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
|---|---|---|---|---|---|---|
| HistGradientBoosting (tuned) | 0.9357 | 0.9243 | 0.9541 | **0.9390** +/- 0.0013 | 0.9872 | 0.9876 |
| RandomForest (tuned) | 0.9344 | 0.9175 | 0.9598 | **0.9382** +/- 0.0011 | 0.9870 | 0.9875 |
| DecisionTree (tuned) | 0.9227 | 0.9097 | 0.9448 | **0.9268** +/- 0.0015 | 0.9788 | 0.9746 |
| LogisticRegression (tuned) | 0.9037 | 0.8623 | 0.9690 | **0.9125** +/- 0.0009 | 0.9695 | 0.9678 |
| Dummy | 0.5184 | 0.5184 | 1.0000 | **0.6828** +/- 0.0000 | 0.5000 | 0.5184 |

## 6. Final model selection (validation evidence only)
- Rule (fixed a priori): highest mean CV F1 among tuned candidates; Dummy never eligible.
- Selected: **HistGradientBoosting (tuned)**
- Alert threshold: **0.480** (max F1 on the internal validation slice of the training partition)

## 7. Test metrics (single pass, untouched testing partition)
| Model | CV F1 (train) | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
| HistGradientBoosting (tuned) | 0.9390 | 0.8712 | 0.8154 | 0.9903 | **0.8944** | 0.9868 | 0.9903 | 0.2747 | 0.0097 |
| RandomForest (tuned) | 0.9382 | 0.8663 | 0.8099 | 0.9894 | **0.8907** | 0.9841 | 0.9881 | 0.2845 | 0.0106 |
| DecisionTree (tuned) | 0.9268 | 0.8601 | 0.8098 | 0.9749 | **0.8847** | 0.9693 | 0.9680 | 0.2806 | 0.0251 |
| LogisticRegression (tuned) | 0.9125 | 0.8100 | 0.7550 | 0.9696 | **0.8490** | 0.9531 | 0.9650 | 0.3854 | 0.0304 |
| Dummy | 0.6828 | 0.5506 | 0.5506 | 1.0000 | **0.7102** | 0.5000 | 0.5506 | 1.0000 | 0.0000 |

- Confusion matrix of the chosen model:
  tn=26,836, fp=10,164, fn=440, tp=44,892

## 8. Error analysis (figure 15)
- False positives 10,164 (27.47% of benign flows); false negatives 440 (0.97% of attacks)
- FP feature medians vs correctly-benign medians: {'dur': {'fp_median': 0.6763965000000001, 'correct_benign_median': 0.06324}, 'rate': {'fp_median': 27.216908, 'correct_benign_median': 654.499234}, 'sbytes': {'fp_median': 774.0, 'correct_benign_median': 1012.0}, 'sttl': {'fp_median': 254.0, 'correct_benign_median': 31.0}, 'dttl': {'fp_median': 252.0, 'correct_benign_median': 29.0}, 'ct_dst_src_ltm': {'fp_median': 3.0, 'correct_benign_median': 2.0}}
- FP protocol share: {'tcp': 0.711, 'udp': 0.288, 'igmp': 0.001}; FP state share: {'FIN': 0.709, 'INT': 0.288, 'REQ': 0.003}
- Family recall-ordered table (test set):
| Family | Total (test) | Missed | Missed-share |
|---|---|---|---|
| Fuzzers | 6,062 | 372 | 6.14% |
| Analysis | 677 | 11 | 1.62% |
| Shellcode | 378 | 3 | 0.79% |
| Exploits | 11,132 | 45 | 0.40% |
| DoS | 4,089 | 6 | 0.15% |
| Reconnaissance | 3,496 | 2 | 0.06% |
| Generic | 18,871 | 1 | 0.01% |
| Backdoor | 583 | 0 | 0.00% |
| Worms | 44 | 0 | 0.00% |
- Hardest family: **Fuzzers**; easiest family: **Backdoor**
- Representative error rows: results/error_examples.csv

## 9. Model interpretation (figure 13)
Permutation importance (mean drop in validation F1 when permuted):
| Rank | Feature | Mean drop in F1 |
|---|---|---|
| 1 | sttl | 0.3748 |
| 2 | dbytes | 0.0228 |
| 3 | sbytes | 0.0202 |
| 4 | ct_srv_dst | 0.0174 |
| 5 | ct_dst_src_ltm | 0.0170 |
| 6 | dmean | 0.0121 |
| 7 | spkts | 0.0108 |
| 8 | ct_srv_src | 0.0107 |
| 9 | service | 0.0097 |
| 10 | dpkts | 0.0084 |
| 11 | dloss | 0.0080 |
| 12 | synack | 0.0067 |
| 13 | sloss | 0.0066 |
| 14 | smean | 0.0047 |
| 15 | dload | 0.0043 |

- Top-5 summary: **sttl** (0.3748), **dbytes** (0.0228), **sbytes** (0.0202), **ct_srv_dst** (0.0174), **ct_dst_src_ltm** (0.0170)
- Limits: predictive importance is statistical, not causal; an important feature does NOT prove malicious intent.

## 10. Leakage sensitivity (companion rows in tables)
- Test rows exactly duplicated in training: 4,247 (5.16%)
- Metrics on ALL test rows: accuracy 0.8712, precision 0.8154, recall 0.9903, F1 0.8944
- Metrics on UNIQUE-ONLY test rows: accuracy 0.8679, precision 0.8090, recall 0.9896, F1 0.8902
- Read as: the drop between the two rows quantifies the inflation from duplicate flows shared across files (release artifact, documented in the limitations chapter).

## 11. Optional multi-class attack-family classifier (figure 16)
- Classes (10): Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms
- Test accuracy: **0.6814**, macro-F1: **0.5129**, weighted-F1: 0.7345
- Classes with recall < 0.5: Analysis (38.26%), DoS (12.40%)

## 12. Demonstration (notebook section 19)
- Final model: HistGradientBoosting (tuned), threshold 0.480
- 10 held-out records (5 malicious, 5 benign, seeded): demo accuracy **80.00%**
- CSVs: results/demo_predictions.csv, results/demo_input_features.csv

## 13. Versions and runtime
- {'python': '3.12.10', 'numpy': '2.5.4', 'pandas': '3.1.0', 'scikit_learn': '1.9.1', 'platform': 'Windows-11-10.0.26300-SP0', 'cpu_count': 20}
- Full run wall time: 6.2 min

## 14. Figure register
- fig01_target_distribution.png - class counts (train/test)
- fig02_attack_categories.png - attack families (train/test counts)
- fig03_feature_histograms.png - histograms of key features (benign vs malicious)
- fig04_boxplots.png - boxplots of visible differences
- fig05_correlation_heatmap.png - correlation matrix
- fig06_missing_and_duplicates.png - missing cells + duplicate counts
- fig07_categorical_composition.png - proto/service/state by attack share
- fig08_roc_curves.png - ROC curves on the test set
- fig09_pr_curves.png - precision-recall curves on the test set
- fig10_confusion_matrix_final.png - confusion matrix of the chosen model (test)
- fig10b_confusion_matrix_normalized.png - class-normalised confusion
- fig11_model_comparison.png - bar chart of test metrics
- fig12_threshold_tradeoff.png - operating-threshold trade-off (validation)
- fig13_permutation_importance.png - permutation importance
- fig14_calibration.png - calibration check of the chosen model
- fig15_recall_by_attack_family.png - recall by family (test)
- fig16_multiclass_confusion.png - attack-family confusion matrix
