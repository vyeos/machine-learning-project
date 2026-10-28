# Viva Preparation - Cybersecurity Intrusion Detection (ECSCI24302)

25+ examiner-style questions with model answers. Every number is quoted from
`results/findings.md` (the single source of truth produced by `python -m src.train`).

## A. Dataset & scope

1. **Which dataset did you use, and where did it come from?**
   UNSW-NB15, the official partitioned release (UNSW Canberra Cyber, 2015). Access
   page: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15. The downloader in
   `src/download_data.py` pulls a byte/row-compatible mirror
   (https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15) that needs no credentials and
   verifies row counts - 175,341 training / 82,332 testing rows.

2. **How big is the dataset?**
   175,341 raw training rows (deduplicated to 107,740) and 82,332 testing rows,
   45 columns per file. Released 10 attack classes including Normal.

3. **What is the target?**
   Binary: benign (0) vs malicious (1). `attack_cat` (family annotation) is used only
   for the optional multi-class extension, never as an input.

4. **Class balance in the training partition?**
   56,000 benign vs 119,341 malicious (68.06% attacks, max:min ratio 2.13); testing is
   37,000 benign vs 45,332 malicious (55.06%).

5. **Did you need to impute missing values?**
   No - the audit verified 0 missing cells in both raw files and no non-finite numeric
   cells. The pipeline still includes a median imputer for safe reuse on other data.

## B. Methodology & leakage control

6. **Explain how you prevented data leakage.**
   (a) The official test partition is untouched and evaluated once; (b) tuning happens
   on the training partition only (stratified 60,000-row subsample); (c) preprocessing
   is inside `Pipeline` and fitted on training data after splitting; (d) `id`, `label`,
   `attack_cat` are excluded from inputs; (e) threshold (0.480) is chosen on a stratified
   20% internal validation slice of the training partition, never on test.

7. **Why drop exact duplicates from the training partition but not the test?**
   67,601 exact duplicate rows in training would double-weight identical flows and
   skew fit (reported figure 6). The testing partition stays exactly as released to
   preserve deployment realism - testing is never cleaned.

8. **Why were some training rows shared with the test file? Is that a problem?**
   The official release itself carries overlapping exact flows: 4,247 test rows
   (5.16%) exactly duplicate training flows. Primarily interpreted via the leakage
   sensitivity check - F1 on all test rows 0.8944 vs 0.8902 on unique-only rows, which
   quantifies the inflation and is documented as a release artifact/limitation.

9. **Hyper-parameter tuning - how?**
   GridSearchCV for LogisticRegression (model__C grid) and DecisionTree (max_depth,
   min_samples_leaf, class_weight), RandomizedSearchCV for RandomForest and
   HistGradientBoosting; cv=3, scoring=F1, on a stratified 60,000-row subsample
   (documented compute constraint). Full specs are in `results/tuning_*.json`.

10. **Best hyper-parameters found?**
    For the selected HistGradientBoosting (tuned): max_iter 300, learning_rate 0.03,
    max_leaf_nodes 63, min_samples_leaf 40, l2_regularization 1.0; CV F1 0.9390 +/- 0.0013.

11. **How did you pick the final model?**
    Fixed a priori rule: highest mean CV F1 among tuned candidates; the dummy baseline
    is never eligible. Evidence used is validation-side only (CV on the training
    partition); the test partition was evaluated exactly once at the end.

12. **How is the alert threshold selected?**
    Maximise F1 on the stratified 20% internal validation slice of the training
    partition (rule fixed a priori). Measured optimum: 0.480. The deployment brief
    emphasises recall; max-F1 is the standard compromise against false alarms.

13. **What preprocessing was applied?**
    Inside one pipeline: median imputer + StandardScaler (for the linear model only)
    + OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=25) on
    proto/service/state. Plus 16 schema-backed engineered features (totals, bytes per
    packet, send/receive ratios, loss rates, log1p twins of six skewed magnitudes,
    ftp/http flags, load_gap). No scaler for tree models.

14. **How did you verify no label/feature attack_cat leakage into inputs?**
    Column-exclusion list enforced in `feature_columns`/`build_preprocessor` - the
    input feature count stays 52 (= 49 numeric + 3 categorical) - verified mismatch-free
    label/attack_cat consistency (0 mismatches) - permutation importance only names real
    flow features.

## C. Results & evaluation

15. **Report full test metrics for the chosen model.**
    Accuracy 0.8712, precision 0.8154, recall 0.9903, F1 0.8944, ROC-AUC 0.9868,
    average precision 0.9903, FPR 27.47%, FNR 0.97%. Confusion: tn=26,836, fp=10,164,
    fn=440, tp=44,892.

16. **How does the model's cross-validation F1 compare to test F1?**
    0.9390 CV vs 0.8944 test - the drop illustrates cross-file distribution shift.

17. **Which attack family was hardest and which easiest?**
    Hardest: Fuzzers (6.14% missed, 372 of 6,062). Easiest: Backdoor (0 missed).

18. **What did the error analysis reveal about FPs?**
    FP medians: dur 0.676 (vs 0.063 correct-benign), rate 27.2 (vs 654.5), sbytes 774
    (vs 1,012), sttl 254 (vs 31), dttl 252 (vs 29), ct_dst_src_ltm 3 (vs 2). Shares: 71%
    TCP, 29% UDP; 71% FIN, 29% INT:068, i.e. long-lived high-TTL flows look attack-like.

19. **Is the FPR acceptable? What would you change?**
    The FPR is 27.47%, which is high but is honestly reported as a limitation. It traces
    to the distribution shift between the official train and test partitions and the
    recall-favouring threshold. Follow-up work would tune thresholds per deployment
    risk profile, use cost-based selection and collect deployment-side calibration data.

## D. Interpretation & robustness

20. **What dominates permutation importance, and why is that a caveat?**
    sttl (0.3748), far above the next (dbytes 0.0228). sttl is partly a schema artifact
    of the synthetic capture (attacker flows in the capture carry a distinctive TTL),
    so it may not generalise beyond UNSW-NB15. We note this as a limitation.

21. **How did you sanity-check probability calibration?**
    Calibration plot (figure 14) shows reliability of the chosen model across bins
    (groups); it is a post-evaluation check, not a selection step.

22. **What did the leakage-sensitivity check show?**
    4,247 exactly duplicated test flows (5.16%); F1 0.8944 -> 0.8902 and recall
    0.9903 -> 0.9896 on the unique-only subset; the drop quantifies inflation.

23. **What did the robustness check show at boundaries?**
    Deterministic end-to-end at random_state=42; primary test metrics stand.

## E. Honesty checks

24. **Which reported numbers were marginally lower than expected? How do you disclose that?**
    The demo accuracy is 80% on 10 held-out records - far below headline test metrics
    but it is a small-sample sanity check, not a validation claim. Also, the FPR of
    27.47% is high - disclosed as a limitation, not hidden.

25. **What is demonstrated in `results/demo_predictions.csv`?**
    10 held-out records (5 malicious, 5 benign, seeded rng from the testing partition),
    with predicted label vs actual for the FINAL model and threshold; demo accuracy 80%.

## F. Honesty limit

26. **How did you verify determinism?**
    Full pipeline is deterministic via random_state=42; every number quoted anywhere
    comes from `results/findings.md` produced by one pipeline run - never estimated.

## G. When something is going wrong

27. **If training loss doesn't converge, how do you know?**
    LogisticRegression is run with max_iter=3000 (increased convergence headroom);
    ConvergenceWarning is watched; warning filter is set to fail loudly in `src/` scripts.

28. **If class distribution shifts between train and test, do you handle it?**
    The train/test distribution shift (68.06% vs 55.06% attacks) underlies the FPR of
    27.47% - analysed and disclosed as a limitation rather than post-hoc fixed.

29. **Which classes does the multi-class classifier get wrong?**
    Analysis (recall 38.26%) and DoS (recall 12.40%) - genuinely hard in UNSW-NB15
    (overlapping families).

30. **Would released test flow duplicates make deployment claims stronger or weaker?**
    Weaker honestly (F1 0.8944 -> 0.8902), and we disclose it via the leakage sensitivity
    row rather than hiding it.
