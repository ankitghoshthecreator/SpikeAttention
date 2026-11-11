# SPIKE-ATTENTION â€” Product Requirements Document (PRD)

**Project:** SPIKE-ATTENTION â€” A Hybrid Spatio-Temporal Attention Spiking Neural Network for Two-Channel ECG Analysis
**Academic subject:** Inferential Statistics and Predictive Analysis (ISPA)
**Document status:** Planning phase. No code, no experiments and no results exist yet. Every statement about performance, significance or efficiency in this repository is a *plan*, never a finding.

---

## 1. Project Overview

SPIKE-ATTENTION is a research-oriented ECG beat-classification system built around one question: *does an attention-augmented Spiking Neural Network (SNN) predict cardiac beat classes on unseen patients better than conventional baselines, and is the difference statistically and practically meaningful?*

The project has two equal pillars:

1. **Predictive Analysis**: a patient-wise, leakage-free ECG classification pipeline (preprocessing, spike encoding, spiking feature extraction, temporal attention, inter-lead attention, classification) compared with four baselines under one protocol.
2. **Inferential Statistics**: a set of OOP Python modules that turn raw experimental outputs into defensible conclusions: assumption checks, hypothesis tests, confidence intervals, effect sizes, multiple-comparison correction, correlation, regression, ROC analysis and efficiency analysis.

**This is primarily an ISPA project.** Deep learning is the predictive component being *studied*; the statistical machinery is the main deliverable.

## 2. Problem Statement

ECG beats carry temporal and cross-lead morphology that distinguishes normal rhythm from arrhythmic beats. Conventional classifiers work, but (a) results on MIT-BIH are frequently inflated by beat-level or record-level leakage, (b) model comparisons are often reported as single numbers without uncertainty, and (c) claims about the efficiency of SNNs are often unsupported by measurement. We want an experiment where comparisons are made on unseen patients, with the correct statistical unit, with uncertainty, and with honest efficiency accounting.

## 3. Research Motivation

- SNNs are event-driven and potentially efficient on suitable hardware; their accuracy on physiological signals relative to standard networks is an open empirical question.
- Attention can capture long-range temporal dependencies and lead interactions, but standard softmax attention is *not* spike-driven. The hybrid trade-off must be measured, not assumed.
- Statistical rigor (paired designs, corrected tests, effect sizes) is the main academic contribution.

## 4. Research Questions and Hypotheses

| ID | Research question | Formal hypotheses (alpha configurable, default 0.05) |
|---|---|---|
| RQ1 | Does ST-SNN differ from each baseline in patient-level predictive performance? | H0: mean paired difference in per-patient balanced accuracy = 0. H1: â‰  0 (two-tailed; the primary family is 4 comparisons: ST-SNN vs SVM, CNN, RNN, plain SNN) |
| RQ2 | Are differences practically meaningful? | H0: \|difference\| < SESOI. Assessed via effect size and CI position relative to SESOI (optional TOST) |
| RQ3 | How reliable/uncertain are performance estimates? | Patient-clustered bootstrap CIs for all headline metrics |
| RQ4 | Do the five models differ overall? | H0: all five models have the same per-patient score distribution (Friedman; repeated-measures ANOVA if assumptions hold) |
| RQ5 | Does attention matter inside the SNN? | ST-SNN vs plain SNN (a member of the RQ1 family; optional ablation with temporal-only and lead-only variants is *secondary/exploratory*) |
| RQ6 | Does ST-SNN offer a computational advantage? | Measured latency, parameters, memory, spike activity; *estimated* energy with a transparent method; relationship to performance via correlation/regression |

**SESOI** (smallest effect size of interest) default: 0.02 absolute difference in balanced accuracy. This is a configurable *assumption*, set before experiments, and must be justified in the final report.

## 5. Objectives

**Primary**
- P1. Build a patient-wise, reproducible ECG classification pipeline on MIT-BIH.
- P2. Implement the full ISPA statistical toolkit as real, tested, OOP Python modules.
- P3. Run all five models under an identical protocol and analyse them with the statistical toolkit.

**Secondary**
- S1. Efficiency analysis (measured vs estimated clearly separated).
- S2. Correlation and regression analysis of cost vs performance and of patient covariates vs performance.
- S3. Exploratory ablations (attention variants), labelled exploratory.

## 6. Dataset Requirements

Source: MIT-BIH Arrhythmia Database, PhysioNet (https://physionet.org/content/mitdb/).

Facts to rely on (verify against the dataset page during implementation M1 and record in `data/metadata/dataset_facts.json`): 48 half-hour two-channel recordings from 47 subjects (records 201 and 202 come from the same subject); sampling rate 360 Hz; beat annotations with symbol codes; most records have MLII as the first lead, while records 102, 104, and 114 do not (records 102 and 104 have V5 as first lead and V2 as second; record 114 has V5 as first lead and MLII as second); four records (102, 104, 107, 217) contain paced beats.

| Requirement | Decision |
|---|---|
| DR1 Access | `wfdb` download into `data/raw/`; no SQL database |
| DR2 Metadata | CSV catalog (`data/metadata/record_catalog.csv`) with record id, patient group id, lead names, sampling rate, paced flag, beat counts per class (computed from data, never typed in) |
| DR3 Label mapping | Beat symbols mapped to **AAMI EC57 classes** N, S, V, F, Q via an explicit table in `configs/label_mapping.yaml`. Symbols are annotation-level *beat* labels, not clinical diagnoses |
| DR4 Unit of splitting | **Patient group** (not record, not beat). Records 201 and 202 share one group id |
| DR5 Paced records | Excluded by default (config flag) following AAMI recommendation; excluded ids logged |
| DR6 Class imbalance | Class-weighted cross-entropy as default; optional weighted sampling; SVM uses `class_weight`. Resampling is applied to *training data only* |
| DR7 Reproducibility | Dataset config (record list, mapping, window length, filters) hashed into every experiment's manifest |

**Record-level vs patient-level splitting.** A record-level split treats each recording as independent. It is leak-prone when one subject contributes more than one record (201/202) and, more generally, does not match the deployment question. A patient-level split guarantees that no subject contributes data to more than one of train/validation/test. We use patient-level (grouped) splitting throughout and assert disjointness in code.

**Limitations of the label mapping (documented, not hidden).** AAMI classes are coarse groupings of beat annotations; rhythm-level diagnoses are not covered; class Q (unknown/paced) is nearly empty after paced exclusion and class F and S are rare; ground-truth annotations come from two cardiologists and include annotation uncertainty; 47 subjects is a small population, and per-class results for rare classes will be unstable. The fraction of subjects who contain each class is reported.

## 7. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Download/verify MIT-BIH, build the record catalog |
| FR2 | Bandpass filter (0.5â€“40 Hz Butterworth, zero-phase, configurable), per-lead normalization (statistics fitted on training patients only), beat-centred segmentation |
| FR3 | Patient-grouped, class-stratified splitting; train/validation/test disjointness assertions |
| FR4 | Spike encoding (default: delta/threshold ONâ€“OFF encoding; parameters configurable and logged) |
| FR5 | Spiking convolutional feature extractor with LIF neurons and surrogate-gradient training |
| FR6 | Temporal attention and inter-lead attention modules |
| FR7 | Baselines: SVM, CNN, RNN (GRU/LSTM), plain SNN (no attention) behind a common `BaseModel` interface |
| FR8 | Training with fixed seeds, class-weighted cross-entropy, early stopping on validation only |
| FR9 | Hyperparameter tuning on validation partitions only, equal budget across models |
| FR10 | Repeated patient-grouped stratified K-fold Ã— multiple seeds with out-of-fold predictions persisted (probabilities, labels, patient ids, beat ids) |
| FR11 | Full ISPA analysis through `ExperimentAnalyzer` (Section 8) |
| FR12 | Efficiency profiling (latency, training time, parameters, peak memory, spike activity, MACs/SynOps) |
| FR13 | Plots (PNG), tables (CSV), machine-readable results (JSON), Markdown report |
| FR14 | CLI (`main.py`) with subcommands for each stage; every stage restartable from stored artifacts |
| FR15 | Automated tests for every statistical class against known textbook/SciPy/statsmodels values |

## 8. Inferential Statistics Requirements

**Statistical units (kept distinct):**

| Level | Role |
|---|---|
| ECG beat/segment | Input to models. **Never** used as an independent observation for inference (strongly correlated within patient) |
| Patient group | **Primary statistical unit.** Per-patient score = mean over seeds of metric on that patient's out-of-fold beats |
| Seed run | Secondary unit (N = number of seeds, small); used for variance-of-training analysis |
| CV fold | Used only with the Nadeauâ€“Bengio corrected resampled t-test, because fold results share training data |

**Paired design.** All models see identical outer folds for a given seed. For patient p, model m: `score[p, m]`. The paired difference `d[p] = score[p, ST-SNN] âˆ’ score[p, baseline]` is computed per patient. Per-patient metric: **balanced accuracy over the classes present in that patient** (macro-recall over present classes). Per-patient macro-F1 is undefined when classes are absent, so it is reported only at pooled level.

**Analysis tiers**

| Tier | Analysis | Purpose |
|---|---|---|
| Primary (confirmatory) | 4 paired comparisons on per-patient balanced accuracy; test chosen by `TestSelector` from assumption checks (paired t or Wilcoxon); Holmâ€“Bonferroni across the family; paired Hedges' g / rank-biserial with bootstrap CI | Answers RQ1, RQ2, RQ5 |
| Secondary | Friedman across 5 models + post-hoc Wilcoxon with Holm; repeated-measures ANOVA only if assumptions pass; pooled macro-F1 with patient-clustered bootstrap CI; seed-level and fold-level (corrected t) analyses; Benjaminiâ€“Hochberg for the secondary family | RQ3, RQ4 |
| Exploratory (flagged) | McNemar on pooled beat-level paired errors (dependence caveat; cluster-aware sensitivity analysis), chi-square tests, one-way ANOVA of ECG features across beat classes (on patient-level means), correlation, regression, ablations | Descriptive insight, not confirmatory claims |

**Test applicability (each documented in code docstrings, `arch.md` and the final report):**

| Test | Used for | Condition |
|---|---|---|
| One-sample t | Is mean per-patient score different from a reference (e.g., chance or a published benchmark value supplied by the user)? | Approx. normal sample; independent patients |
| Independent two-sample t / Welch | Compare *different* patient groups (e.g., patients with vs without V beats) or features across independent groups; Welch is the default when variances differ | Independent groups; Levene result guides choice |
| Paired t | Primary comparison when differences pass normality/outlier checks | Differences approx. normal; patients independent |
| Wilcoxon signed-rank | Paired non-parametric primary comparison when normality is doubtful | Symmetric differences; handles ties (zero-difference method documented) |
| McNemar | Pooled paired beat-level correctness, two models | Exploratory; dependence among beats noted; exact binomial when discordant < 25 |
| Chi-square | (a) class distribution independence across train/val/test; (b) association of model correctness with class | Expected counts â‰¥ 5 else Fisher/merge categories |
| One-way ANOVA | Patient-level feature means across beat classes | Normality, homogeneity; else Kruskalâ€“Wallis fallback |
| Repeated-measures ANOVA | 5 models on the same patients | Sphericity assessment or Greenhouseâ€“Geisser correction |
| Friedman | 5 models on the same patients, rank-based | Complete blocks; post-hoc with correction |

**Required outputs per test:** test statistic, degrees of freedom, p-value, decision at alpha, critical region/critical value where defined, effect size, confidence interval, assumption-check summary, plain-language interpretation, plus a **power / minimum detectable effect** statement (`HypothesisTester.power_analysis`) and an explicit Type I / Type II discussion.

**Confidence intervals:** Wilson (single proportions, e.g., accuracy/sensitivity/specificity when treated as binomial, with the dependence caveat), normal approximation (where justified), **patient-clustered percentile/BCa bootstrap** (default for pooled metrics such as macro-F1 and for performance differences).

**Effect sizes:** Cohen's d, paired Cohen's d (d_z), Hedges' g (with small-sample correction), rank-biserial correlation (Wilcoxon), Kendall's W (Friedman), CramÃ©r's V (chi-square), eta-squared / partial eta-squared (ANOVA).

**Correlation:** Pearson, Spearman, correlation matrices with significance and multiple-comparison correction; applied to ECG patient-level features, performance metrics, and cost-vs-performance (small n warned).

**Regression:** simple and multiple OLS (statsmodels), coefficients with CIs, RÂ², adjusted RÂ², F-test, residual diagnostics (normality, Breuschâ€“Pagan, Durbinâ€“Watson, VIF, Cook's distance). Intended use: explain per-patient score by covariates (model, mean heart rate, abnormal-beat fraction, noise level) and cost-vs-performance across model configurations. Limitations: few model-level points, repeated measures per patient violate independence (cluster-robust standard errors or mixed model noted as optional), non-linearity, associations are not causal.

**Multiple comparison:** Bonferroni, Holmâ€“Bonferroni, Benjaminiâ€“Hochberg; the family definition (primary vs secondary) is fixed in configuration *before* running experiments.

**Assumption testing:** Shapiroâ€“Wilk (plus Q-Q plots; a non-significant result is never reported as proof of normality), Levene/Brownâ€“Forsythe, outlier screening (IQR rule, robust z-scores), independence discussion, sphericity.

## 9. Predictive Analysis Requirements

- Outer evaluation: repeated, patient-grouped, class-stratified K-fold (default K = 5, seeds default 5; both configurable). Stratification at group level uses `StratifiedGroupKFold`; imperfect class balance per fold is reported.
- Inner validation: a patient-wise subset of each training fold for early stopping and hyperparameter selection. Test patients are never used for any fitting, normalization, thresholding or tuning.
- Hyperparameter tuning: small, equal-budget search per model (random/grid) on validation only.
- Persisted out-of-fold predictions enable: prediction-probability analysis (calibration curves, confidence distribution for correct vs wrong), generalization analysis (train/validation/test gap, per-patient score distributions), and all inferential analyses.
- Metrics: TP/TN/FP/FN (one-vs-rest), accuracy, precision, recall/sensitivity, specificity, F1, macro-F1, weighted-F1, classification report, confusion matrix.
- ROC: one-vs-rest ROC Curve per class; macro-averaged **AU (Area Under the ROC Curve)**; threshold analysis (Youden index); documented behaviour when a class is absent.

## 10. Model Requirements

| Model | Input | Notes |
|---|---|---|
| SVM | Handcrafted/compact features (wavelet or morphology + RR features), both leads | RBF kernel; features fitted on training only |
| CNN | Filtered 2-channel beat window | 1-D CNN, matched parameter budget (within a documented tolerance) |
| RNN | Filtered 2-channel beat window | GRU/LSTM, matched budget |
| Plain SNN | Spike-encoded beat | Spiking conv + LIF, global pooling, no attention |
| **ST-SNN (proposed)** | Spike-encoded beat | Spiking conv + LIF â†’ temporal attention + inter-lead attention â†’ pooling â†’ FC â†’ softmax |

Model-requirement notes:
- LIF dynamics: `Ï„ dV/dt = âˆ’(V âˆ’ V_reset) + RÂ·I(t)`, spike if `V > V_th`, then `V â† V_reset`. Discretized with Euler step; parameters (Ï„, V_th, V_reset, R, Î”t) are configured and logged.
- Surrogate gradient: arctan surrogate by default.
- Attention: `Attention(Q,K,V) = softmax(QKáµ€/âˆšd_k)V`, applied on real-valued features derived from the spiking layer (firing-rate or membrane-potential trace). The model is therefore a **hybrid**, not fully spike-driven; this limits any hardware-efficiency claim and is documented.
- Keep the model small; the statistical pipeline must work first.

## 11. ISPA Syllabus-to-Feature Mapping

| ISPA Concept | Python Module | Class | Application in Project | Output |
|---|---|---|---|---|
| Descriptive statistics | `statistics/descriptive_statistics.py` | `DescriptiveStatistics` | Class distribution, ECG feature distributions, metric summaries (mean, median, mode, variance, SD, quartiles, IQR, range) | `descriptive_*.csv`, histograms, box/violin plots |
| Hypothesis testing framework | `statistics/hypothesis_testing.py`, `statistics/base.py` | `HypothesisTester`, `BaseStatisticalTest`, `TestResult` | H0/H1, alpha, critical region, one/two-tailed, Type I/II error, power | `hypothesis_results.json/csv` |
| t-tests | `statistics/ttest.py` | `OneSampleTTest`, `IndependentTTest`, `WelchTTest`, `PairedTTest`, `CorrectedResampledTTest` | Paired model comparison; group comparisons; fold-level corrected test | test result records, p-value chart |
| Chi-square | `statistics/chi_square.py` | `ChiSquareTest` | Split-distribution check; correctness-by-class association | contingency tables, results |
| ANOVA | `statistics/anova.py` | `OneWayANOVA`, `RepeatedMeasuresANOVA` | Feature-by-class; five-model comparison | ANOVA tables |
| Non-parametric tests | `statistics/nonparametric_tests.py` | `WilcoxonSignedRankTest`, `FriedmanTest`, `KruskalWallisTest` | Robust paired and multi-model comparison | results + post-hoc table |
| McNemar | `statistics/mcnemar_test.py` | `McNemarTest` | Paired pooled error comparison (exploratory) | 2Ã—2 discordance tables |
| Confidence intervals | `statistics/confidence_intervals.py` | `ConfidenceIntervalEstimator` | Wilson, normal, clustered bootstrap for metrics and differences | CI tables, CI plots |
| Effect size | `statistics/effect_size.py` | `EffectSizeAnalyzer` | Cohen's d, d_z, Hedges' g, rank-biserial, Kendall's W | effect-size table/plot |
| Correlation | `statistics/correlation.py` | `CorrelationAnalyzer` | Pearson/Spearman, matrices, significance | heatmaps, tables |
| Regression | `statistics/regression.py` | `RegressionAnalyzer` | Simple/multiple OLS, diagnostics, cost-vs-performance | coefficient tables, regression/residual plots |
| Predictive pipeline | `evaluation/cross_validation.py`, `prediction/predictor.py`, `evaluation/model_evaluator.py`, `models/baselines/*`, `training/hpo.py` | `PatientGroupCrossValidator`, `Predictor`, `ModelEvaluator`, `HyperparameterTuner` | Patient-wise CV, baselines, tuning, probability analysis | OOF prediction files, metric tables |
| Confusion matrix / metrics | `statistics/classification_metrics.py` | `ClassificationMetrics` | TP/TN/FP/FN, accuracy, precision, recall, specificity, F1, macro/weighted F1 | report tables, confusion-matrix PNG |
| ROC analysis | `statistics/roc.py` | `ROCAnalyzer` | One-vs-rest ROC Curve, TPR/FPR, thresholds, AU (Area Under the ROC Curve) | ROC PNG, threshold CSV |
| Multiple comparison | `statistics/multiple_comparison.py` | `MultipleComparisonCorrector` | Bonferroni, Holm, Benjaminiâ€“Hochberg over defined families | adjusted p-value tables |
| Assumption testing | `statistics/assumption_tests.py` | `AssumptionTester` | Shapiroâ€“Wilk, Levene, outliers, Q-Q, sphericity | assumption report, Q-Q plots |
| Computational efficiency | `statistics/efficiency_analysis.py`, `evaluation/efficiency_profiler.py` | `EfficiencyAnalyzer`, `EfficiencyProfiler` | Latency, parameters, memory, spike activity, estimated energy | efficiency tables/plots |
| Workflow integration | `statistics/experiment_analyzer.py`, `statistics/test_selector.py` | `ExperimentAnalyzer`, `TestSelector` | Orchestrates all analyses on stored results | final JSON, Markdown report |

**Excluded or optional techniques (with reasons):** Time-series tests on single recordings (segments are autocorrelated, not independent samples); paired t-tests on raw beat-level scores (pseudoreplication); mixed-effects models (optional extension, outside core syllabus); Bayesian comparison (out of scope); independent two-sample t-test on model scores from the same patients (design is paired, so it is used only for genuinely independent groups).

## 12. OOP Requirements

- Abstract bases only where they remove duplication: `BaseStatisticalTest` (template method: `validate_input â†’ check_assumptions â†’ compute â†’ interpret`), `BaseModel` (`fit/predict_proba/predict/count_parameters`), `BaseSpikeEncoder`, `BaseAnalyzer` (common `run()`/`to_dict()` contract).
- Encapsulation: result objects (`TestResult`, `IntervalResult`, `EffectSizeResult`) are immutable dataclasses.
- Polymorphism: `ExperimentAnalyzer` treats every test as a `BaseStatisticalTest`; every model as a `BaseModel`.
- Composition: `ExperimentAnalyzer` *has* an `AssumptionTester`, `TestSelector`, `HypothesisTester`, `ConfidenceIntervalEstimator`, `EffectSizeAnalyzer`, `MultipleComparisonCorrector`, etc.; `ModelEvaluator` has a `ClassificationMetrics` and `ROCAnalyzer`.
- No artificial inheritance hierarchies.

## 13. User Workflows

1. **Setup**: create environment â†’ `python main.py data download` â†’ `data catalog`.
2. **Preprocess**: `python main.py data preprocess --config configs/dataset.yaml`.
3. **Run experiments**: `python main.py experiment run --config configs/experiment.yaml` (all models, folds, seeds; resumable).
4. **Analyse**: `python main.py analyze --run-id <id>` â†’ statistics, figures, report.
5. **Inspect**: open `results/<run_id>/reports/report.md`.
6. **Reproduce**: rerun with the stored `manifest.json` (config hash, seeds, package versions, git commit).

## 14. Inputs and Outputs

**Inputs:** YAML configs; PhysioNet MIT-BIH files; optional user-provided reference benchmark values.
**Outputs (per run):** `manifest.json`; `predictions/*.parquet|csv` (out-of-fold probabilities); `metrics/*.csv`; `statistics/*.json|csv`; `figures/*.png`; `reports/report.md`; `logs/*.log`.

## 15. Non-Functional Requirements

NFR1 Python 3.11, runs locally; CPU is sufficient for statistics, optional GPU for deep models. NFR2 Deterministic where feasible (seeded Python/NumPy/PyTorch; non-deterministic GPU ops documented). NFR3 No SQL. NFR4 Every module unit-tested; statistical classes verified against SciPy/statsmodels references. NFR5 Typed, documented public APIs. NFR6 Structured logging. NFR7 Clear failure messages when assumptions or input shapes are violated. NFR8 No hard-coded results anywhere (enforced by a test that scans for numeric literals in report templates).

## 16. Evaluation Criteria

- Correctness: leakage assertions pass; statistical classes agree with reference implementations to numerical tolerance.
- Academic quality: every inference states unit, pairing, assumptions, effect size, CI, corrected p-value.
- Honesty: negative/inconclusive outcomes reported in the same format as positive ones.
- Reproducibility: a fresh clone plus configs reproduces the same tables up to documented tolerance.

## 17. Experimental Methodology (summary)

1. Load, filter, normalize (train-fitted), segment beats (default 1 s window centred on annotated R-peak; configurable).
2. Patient-group assignment (201 and 202 merged); optional paced-record exclusion.
3. For each seed: patient-grouped stratified K-fold; inner patient-wise validation split.
4. Tune on validation; fit; predict out-of-fold probabilities.
5. Compute per-patient, per-fold, per-seed and pooled metrics.
6. Efficiency profiling on a fixed hardware/software configuration (hardware recorded).
7. `ExperimentAnalyzer` runs assumption checks â†’ test selection â†’ tests â†’ CIs â†’ effect sizes â†’ corrections â†’ interpretation â†’ plots â†’ report.

**Selection rule (documented, pre-specified):** for each primary comparison, Shapiroâ€“Wilk and outlier screening on the paired differences plus a Q-Q inspection decide between paired t and Wilcoxon; the decision rule is fixed in `configs/analysis.yaml` before running. Both results are stored; only the pre-specified one enters the confirmatory family.

## 18. Risks and Limitations

| Risk | Mitigation |
|---|---|
| Small population (â‰ˆ43â€“47 patient groups) â†’ low power, unstable rare-class estimates | Report power/MDE; patient-clustered CIs; avoid rare-class claims |
| Severe class imbalance | Weighted loss, balanced-accuracy primary metric, per-class reporting |
| Label coarse-graining (AAMI) | Explicit limitation section; no claim of diagnostic validity |
| SNN training instability | Small architecture first; surrogate-gradient tuning; fallback encoders |
| Hybrid attention not event-driven | Report which ops are spike-driven; limit energy claims |
| Non-comparable baselines | Matched parameter budgets, equal tuning budget, same folds |
| Multiple-testing inflation | Pre-registered families, Holm/BH |
| Fold-result dependence | Patient-level primary unit; corrected resampled t for folds |
| Energy claims | Measured vs estimated separated; estimation constants configurable and cited as assumptions |
| MIT-BIH is a historical, single-source dataset | No external-validity claims; future work on additional databases |

## 19. Success Criteria

- Pipeline runs end to end from raw data to report with one command sequence.
- No leakage (asserted in tests and logged per fold).
- All ISPA modules implemented, tested and exercised by `ExperimentAnalyzer` on real outputs.
- The report states each conclusion with test, assumptions, effect size, CI, corrected p and practical-significance judgment, **whichever way the result goes**.

## 20. Development Milestones

| # | Milestone | Key deliverables |
|---|---|---|
| M0 | Scaffold & environment | Repo, configs, CI-style test run, seed/logging utils |
| M1 | Data ingestion | Download, catalog, label mapping, dataset facts |
| M2 | Preprocessing & splitting | Filters, normalization, segmentation, `PatientSplitter`, leakage tests |
| M3 | Core ISPA modules (part 1) | Descriptive, assumption tests, base test classes, t-tests, effect size, CIs, multiple comparison, with tests against SciPy/statsmodels |
| M4 | Core ISPA modules (part 2) | Non-parametric, McNemar, chi-square, ANOVA, correlation, regression, metrics, ROC, `ExperimentAnalyzer` (tested with synthetic *test fixtures* only) |
| M5 | Baselines | SVM, CNN, RNN with common interface |
| M6 | Spike encoding & plain SNN | Encoder, LIF, surrogate gradient, plain SNN |
| M7 | ST-SNN | Temporal and inter-lead attention |
| M8 | Training/CV pipeline | Trainer, HPO, repeated grouped CV, OOF storage |
| M9 | Real experiments | Full runs, efficiency profiling |
| M10 | Analysis & reporting | Plots, Markdown report, documentation update |

## 21. Out of Scope

SQL/databases; real-time or clinical deployment; hardware neuromorphic deployment; other ECG databases (future work); diagnostic claims; web UI; federated learning; mixed-effects/Bayesian modelling (optional later).
