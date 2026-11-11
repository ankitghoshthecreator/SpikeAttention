# ISPA Syllabus and SPIKE-ATTENTION Mapping

> **Evidence and scope.** This document maps the ISPA topics listed in the task context to code that exists in this repository. No separate university syllabus file was present in the repository; therefore the candidate topic list supplied in the task is treated as the syllabus scope. Topics outside that list are not added merely because the code or a library could support them. “Actively used” means that the completed `infra_pilot_cuda` execution path invokes the method for the primary result; “implemented but not used” means source code exists but it is not part of the primary inferential pipeline.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Syllabus-to-Project Mapping](#2-syllabus-to-project-mapping)
3. [Detailed Explanation of Each Common Topic](#3-detailed-explanation-of-each-common-topic)
4. [Statistical Methods and Equations](#4-statistical-methods-and-equations)
5. [Practical Application in the Project](#5-practical-application-in-the-project)
6. [Actual Experimental Results](#6-actual-experimental-results)
7. [Interpretation](#7-interpretation)
8. [Viva Preparation](#8-viva-preparation)

## 1. Introduction

In this project, **ISPA** refers to the syllabus concepts in *Introduction to Statistical and Predictive Analysis* as they apply to the statistical evaluation of a machine-learning experiment. The project classifies ECG heartbeats into five AAMI classes: **N, S, V, F, and Q**. It compares five classifiers: `svm` (implemented as a calibrated linear SGD margin classifier), `cnn`, `rnn`, `plain_snn`, and the proposed `st_snn`.

Predictive analysis answers how well a fitted model classifies new beats. Statistical analysis answers a different question: whether measured differences across patients are large and consistent enough to support a comparison. Predictive metrics such as accuracy and balanced accuracy summarize predictions; paired tests, confidence intervals, and effect sizes quantify uncertainty across the patient groups. Both are needed: a high beat-level score alone does not establish that one model generalizes better across patients.

The experiment keeps train, validation, and test partitions separate. Validation results guide checkpoint selection for neural models. Held-out test predictions provide final fold performance. For primary inference, the independent unit is the patient group: each model is compared on the same patients. Thousands of beats from one person do not count as thousands of independent experimental subjects.

## 2. Syllabus-to-Project Mapping

| Syllabus Topic | Project Implementation | Source File | Actual Usage |
|---|---|---|---|
| Data Preparation and Preprocessing | MIT-BIH record loading, band-pass filtering, beat segmentation, label mapping, and patient-group splits | `src/spikeattn/data/`; `src/spikeattn/preprocessing/`; `configs/dataset.yaml`; `configs/label_mapping.yaml`; `experiments/run_experiment.py` | Actively used in the primary experiment |
| Hypothesis Testing | Two-sided paired model comparisons; assumption-based choice between paired t and Wilcoxon; matched Friedman omnibus test; Holm adjustment | `src/spikeattn/statistics/experiment_analyzer.py`; `test_selector.py`; `ttest.py`; `nonparametric_tests.py`; `multiple_comparison.py`; `configs/analysis.yaml` | Actively used in the primary experiment |
| Paired T-Test | Patient-matched differences in balanced accuracy for ST-SNN versus each baseline | `src/spikeattn/statistics/ttest.py`; `src/spikeattn/statistics/test_selector.py`; `src/spikeattn/statistics/experiment_analyzer.py` | Actively used in the primary experiment when selection assumptions permit; all four pilot comparisons selected it |
| ANOVA | One-way and repeated-measures ANOVA implementations | `src/spikeattn/statistics/anova.py` | Implemented but not used in the primary experiment |
| Chi-Square Test | Chi-square test utility for categorical frequency data | `src/spikeattn/statistics/chi_square.py` | Implemented but not used in the primary experiment |
| Correlation | Pearson/Spearman association helpers; analyzer’s exploratory cost/performance relationship analysis | `src/spikeattn/statistics/correlation.py`; `src/spikeattn/statistics/experiment_analyzer.py` | Implemented but not used in the primary model-comparison claim; available in exploratory relationship analysis |
| Regression | OLS regression helper and diagnostics; analyzer’s exploratory cost/performance relationship analysis | `src/spikeattn/statistics/regression.py`; `src/spikeattn/statistics/experiment_analyzer.py` | Implemented but not used in the primary model-comparison claim; available in exploratory relationship analysis |
| Descriptive Statistics | Patient-level balanced-accuracy summaries by model | `src/spikeattn/statistics/descriptive_statistics.py`; `src/spikeattn/statistics/experiment_analyzer.py` | Actively used in the primary experiment |
| Confidence Intervals | Patient-level percentile bootstrap intervals for each model and paired bootstrap intervals for model differences; paired t intervals | `src/spikeattn/statistics/confidence_intervals.py`; `src/spikeattn/statistics/experiment_analyzer.py`; `src/spikeattn/statistics/ttest.py` | Actively used in the primary experiment |
| Predictive Analysis | Train models, choose neural checkpoints on validation data, predict held-out test beats, and compute classification metrics | `experiments/run_experiment.py`; `src/spikeattn/evaluation/model_evaluator.py`; `src/spikeattn/statistics/classification_metrics.py` | Actively used in the primary experiment |

## 3. Detailed Explanation of Each Common Topic

The ten topics below are included because each appears in the task-provided syllabus candidate list and has a corresponding project implementation. ANOVA and chi-square are not described as pilot analyses: their implementation is present, but the primary analyzer does not call them.

### 3.1 Data Preparation and Preprocessing

1. **Terminology:** Data preparation, filtering, segmentation, label mapping, grouped partitioning.
2. **Definition:** Transforming raw ECG recordings into consistently labeled heartbeat examples while preserving subject identity for unbiased evaluation.
3. **Mathematical formulation:** For a discrete filter, (y[n]=\sum_{k=0}^{M}b_kx[n-k]-\sum_{k=1}^{N}a_ky[n-k]). A beat centered at annotation index (r_i) is represented by samples (x[r_i-L_-,\ldots,r_i+L_+]), where the configured window has 360 samples at 360 Hz (one second).
4. **Variable definitions:** (x[n]) is the input ECG signal, (y[n]) the filtered signal, (a_k,b_k) filter coefficients, (r_i) the annotated beat location, and (L_-,L_+) the samples on either side of the center.
5. **Purpose in the syllabus:** Show how raw observations become valid analysis inputs, including the importance of consistent units, labels, and sampling.
6. **Application in SPIKE-ATTENTION:** MIT-BIH records are filtered with a configured Butterworth band-pass (0.5–40 Hz, order 4), segmented around annotated beats, and mapped to fixed classes N/S/V/F/Q. Patient groups are kept together in splits. The configured test data do not train the preprocessing transforms.
7. **Relevant source files:** `configs/dataset.yaml`, `configs/label_mapping.yaml`, `src/spikeattn/preprocessing/`, `src/spikeattn/data/`, and `experiments/run_experiment.py`.
8. **Advantages:** Standardizes inputs, focuses each sample on a beat, and patient-group partitioning reduces subject leakage.
9. **Disadvantages and limitations:** Class imbalance remains; filtering cannot correct noisy or mislabeled records. Some rare labels may be absent in an individual fold. The project’s lead normalizer exists but is not called by the current experiment path; the SVM feature scaler is fit within its training pipeline.
10. **Practical example using the project's ECG data:** An annotation at an R-peak identifies a beat; the pipeline takes the configured one-second segment from both leads, applies the label map, and records its patient group so every beat from that patient is assigned consistently.

### 3.2 Hypothesis Testing

1. **Terminology:** Null hypothesis, alternative hypothesis, significance level, test statistic, p-value, multiplicity adjustment.
2. **Definition:** A procedure for assessing how compatible observed data are with a specified null model. It does not prove a hypothesis true or false.
3. **Mathematical formulation:** For each baseline, (H_0: \Delta=0) against the two-sided (H_1: \Delta\ne0), where (\Delta) is the population mean patient-level difference in balanced accuracy. The decision threshold is configured as (\alpha=0.05); the primary family has four comparisons and uses Holm adjustment.
4. **Variable definitions:** (H_0,H_1) are hypotheses; (\Delta) is the paired difference parameter; (\alpha) is the preselected Type I error threshold.
5. **Purpose in the syllabus:** Distinguish evidence against a null model from descriptive differences and explain controlled decision rules.
6. **Application in SPIKE-ATTENTION:** The analyzer aligns patient scores before comparison, checks the differences, selects paired t or Wilcoxon, computes a matched Friedman secondary omnibus test, and corrects the four ST-SNN comparisons with Holm. The pilot’s four selected tests were paired t-tests.
7. **Relevant source files:** `src/spikeattn/statistics/experiment_analyzer.py`, `test_selector.py`, `multiple_comparison.py`, `nonparametric_tests.py`, and `configs/analysis.yaml`.
8. **Advantages:** Makes the comparison explicit and controls false-positive risk when multiple baselines are tested.
9. **Disadvantages and limitations:** A p-value does not measure effect importance; small patient counts yield low power. A nonsignificant result is not proof of equivalence. Data-dependent test selection and small samples make assumption checks imperfect.
10. **Practical example using the project's ECG data:** For each of nine test patients, subtract the selected baseline’s balanced accuracy from ST-SNN’s score, then assess whether the paired differences support a nonzero population difference. Apply Holm across the four baseline questions.

### 3.3 Paired T-Test

1. **Terminology:** Paired samples, within-patient difference, degrees of freedom.
2. **Definition:** A one-sample t-test on the differences between two measurements made on matched units. Here, each unit is a patient group assessed by both models.
3. **Mathematical formulation:**
   \[
   t=\frac{\bar d}{s_d/\sqrt n},\qquad df=n-1.
   \]
4. **Variable definitions:** (d_i=BA_{ST,i}-BA_{B,i}); (BA_{ST,i}) and (BA_{B,i}) are ST-SNN and baseline balanced accuracy for patient (i); (\bar d) is the mean difference; (s_d) is its sample standard deviation; (n) is the number of matched patients.
5. **Purpose in the syllabus:** Compare two related means while accounting for within-subject pairing.
6. **Application in SPIKE-ATTENTION:** The pilot’s paired patient scores selected paired t-tests for ST-SNN against `svm`, `cnn`, `rnn`, and `plain_snn`; each comparison had (n=9) and (df=8). The test selector uses paired t when its normality/outlier rule permits, otherwise Wilcoxon.
7. **Relevant source files:** `src/spikeattn/statistics/ttest.py`, `test_selector.py`, and `experiment_analyzer.py`.
8. **Advantages:** Removes some between-patient variability and directly tests the average within-patient difference.
9. **Disadvantages and limitations:** Requires matched patient observations and is sensitive to severe non-normality/outliers in differences. Nine groups provide limited precision. Pairing by beat would be pseudoreplication and is not the primary analysis.
10. **Practical example using the project's ECG data:** If a patient’s ST-SNN balanced accuracy is 0.40 and the same patient’s RNN score is 0.55, the paired difference is (-0.15); the t-test analyzes the nine such matched differences.

### 3.4 ANOVA

1. **Terminology:** Analysis of variance, between-group variation, within-group variation, F statistic.
2. **Definition:** A family of tests comparing means across multiple groups; repeated-measures ANOVA accounts for repeated outcomes on the same subjects.
3. **Mathematical formulation:** For one-way independent groups, (F=MS_B/MS_W), with (MS_B=SS_B/(k-1)) and (MS_W=SS_W/(N-k)). The source also has repeated-measures functionality; for a balanced repeated-measures design, its treatment effect is based on treatment variation relative to residual variation after subject effects are accounted for.
4. **Variable definitions:** (SS_B,SS_W) are between- and within-group sums of squares; (MS_B,MS_W) are their mean squares; (k) is number of groups; (N) is total observations.
5. **Purpose in the syllabus:** Test whether at least one group mean differs when comparing more than two groups.
6. **Application in SPIKE-ATTENTION:** ANOVA classes exist in `anova.py`, but the primary analyzer uses paired model comparisons plus Friedman, not ANOVA. The pilot report contains no ANOVA result.
7. **Relevant source files:** `src/spikeattn/statistics/anova.py`.
8. **Advantages:** A single omnibus test can assess several group means without starting with many pairwise tests.
9. **Disadvantages and limitations:** Independent one-way ANOVA would ignore that model scores are matched by patient. Repeated-measures ANOVA needs suitable assumptions, including distributional and covariance conditions. The project does not use this implementation for its primary comparison.
10. **Practical example using the project's ECG data:** A hypothetical analysis might compare the mean patient balanced accuracies across all five models with repeated measures. That is an example of possible use, not a method used for `infra_pilot_cuda`.

### 3.5 Chi-Square Test

1. **Terminology:** Observed frequency, expected frequency, contingency table, degrees of freedom.
2. **Definition:** A test comparing categorical frequency counts to expected counts under a null hypothesis, such as independence.
3. **Mathematical formulation:**
   \[
   \chi^2=\sum_{i=1}^{R}\sum_{j=1}^{C}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}.
   \]
4. **Variable definitions:** (O_{ij}) is the observed count in row (i), column (j); (E_{ij}) is the expected count under the null; (R,C) are table dimensions. For an independence test, expected count (E_{ij}=(\text{row}_i\text{ total})(\text{column}_j\text{ total})/N).
5. **Purpose in the syllabus:** Evaluate association between categorical variables or goodness-of-fit of observed categorical frequencies.
6. **Application in SPIKE-ATTENTION:** A chi-square test utility is implemented but is not called by the primary experiment analyzer. The project’s main inference is on matched patient balanced accuracy; the pilot does not report a chi-square result.
7. **Relevant source files:** `src/spikeattn/statistics/chi_square.py`.
8. **Advantages:** Applies to categorical frequency data and can summarize association in a contingency table.
9. **Disadvantages and limitations:** Expected-count conditions matter. Beat-level predictions from the same patient are dependent, so a naive chi-square test treating each heartbeat as independent would overstate information. No such test is used in this pilot.
10. **Practical example using the project's ECG data:** A class-by-predicted-class confusion table contains categorical counts, but applying a population test to all beat counts would require an independence strategy accounting for patient clustering. The current project reports the confusion matrices descriptively instead.

### 3.6 Correlation

1. **Terminology:** Correlation coefficient, rank correlation, monotonic association.
2. **Definition:** A numerical summary of the strength and direction of association between two variables; it does not establish causation.
3. **Mathematical formulation:** Pearson correlation is (r=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i(x_i-\bar x)^2\sum_i(y_i-\bar y)^2}}). Spearman’s coefficient is the Pearson correlation of ranks: (\rho_s=\operatorname{corr}(\operatorname{rank}(x),\operatorname{rank}(y))).
4. **Variable definitions:** (x_i,y_i) are paired observations; (\bar x,\bar y) are sample means; ranks replace raw values for Spearman.
5. **Purpose in the syllabus:** Describe whether two numeric measurements vary together, linearly (Pearson) or monotonically (Spearman).
6. **Application in SPIKE-ATTENTION:** Correlation helpers are implemented and the analyzer includes exploratory Spearman relationships between efficiency and performance. These are not the primary ST-SNN superiority tests; report only a relationship result if the saved analysis artifact contains one.
7. **Relevant source files:** `src/spikeattn/statistics/correlation.py`; `src/spikeattn/statistics/experiment_analyzer.py`.
8. **Advantages:** Compactly describes direction and strength and can highlight potential efficiency/performance trade-offs.
9. **Disadvantages and limitations:** Correlation can be driven by outliers or confounding, may miss nonlinear relationships, and does not imply causality. With five model-level points, exploratory association estimates are unstable.
10. **Practical example using the project's ECG data:** An exploratory question is whether models with lower measured inference latency tend to have higher patient-level balanced accuracy. Such an association would not show that latency causes accuracy.

### 3.7 Regression

1. **Terminology:** Response, predictor, coefficient, residual, ordinary least squares (OLS).
2. **Definition:** A model that describes a conditional relationship between a response variable and one or more predictors.
3. **Mathematical formulation:** For simple linear regression, (y_i=\beta_0+\beta_1x_i+\epsilon_i). OLS chooses coefficients minimizing (\sum_i(y_i-\beta_0-\beta_1x_i)^2).
4. **Variable definitions:** (y_i) is response (for example balanced accuracy), (x_i) a predictor (for example latency), (\beta_0) intercept, (\beta_1) slope, and (\epsilon_i) residual.
5. **Purpose in the syllabus:** Estimate and describe how a response changes with predictors, while making model assumptions explicit.
6. **Application in SPIKE-ATTENTION:** An OLS helper and exploratory efficiency/performance relationship path are implemented. This regression is not the primary evidence for model superiority and should not be treated as a causal model.
7. **Relevant source files:** `src/spikeattn/statistics/regression.py`; `src/spikeattn/statistics/experiment_analyzer.py`.
8. **Advantages:** Provides an interpretable slope and can quantify an observed trade-off.
9. **Disadvantages and limitations:** Linearity, residual, and sample-size assumptions matter. A few model aggregates cannot support a robust causal conclusion; model identity and hardware differ together in this pilot.
10. **Practical example using the project's ECG data:** Fit balanced accuracy as (y) and inference latency as (x) across model summaries to explore whether faster inference accompanies lower or higher accuracy. The pilot’s mixed CPU/GPU execution makes this descriptive and hardware-confounded.

### 3.8 Descriptive Statistics

1. **Terminology:** Mean, median, standard deviation, range, quartiles, sample size.
2. **Definition:** Numerical summaries describing observed data without generalizing through a hypothesis test.
3. **Mathematical formulation:** (\bar x=\frac1n\sum_i x_i); (s=\sqrt{\frac{1}{n-1}\sum_i(x_i-\bar x)^2}). The median and quartiles summarize location and spread robustly.
4. **Variable definitions:** (x_i) is one patient’s model score; (n) is the number of patient observations; (\bar x) and (s) are sample mean and standard deviation.
5. **Purpose in the syllabus:** Describe distribution, central tendency, spread, and data coverage before inferential claims.
6. **Application in SPIKE-ATTENTION:** The analyzer summarizes per-patient balanced accuracy by model; beat-level classification metrics are separately stored for prediction evaluation.
7. **Relevant source files:** `src/spikeattn/statistics/descriptive_statistics.py`; `src/spikeattn/statistics/experiment_analyzer.py`.
8. **Advantages:** Makes model differences and patient variability understandable and provides context for tests.
9. **Disadvantages and limitations:** A mean alone hides skew, outliers, class-specific failure, and sample size. Descriptive differences do not establish significance.
10. **Practical example using the project's ECG data:** The pilot mean patient balanced accuracy was 0.4103 for ST-SNN and 0.5290 for CNN; the nine patient scores vary, so this difference must be interpreted with paired tests and intervals.

### 3.9 Confidence Intervals

1. **Terminology:** Confidence level, interval estimate, bootstrap resampling, paired interval.
2. **Definition:** An interval procedure that, under repeated sampling and its assumptions, covers the target parameter at the stated long-run rate.
3. **Mathematical formulation:** A paired t interval is (\bar d\pm t_{1-\alpha/2,n-1}s_d/\sqrt n). A percentile bootstrap interval uses quantiles (q_{\alpha/2}(\hat\theta^*)) and (q_{1-\alpha/2}(\hat\theta^*)) from resampled estimates.
4. **Variable definitions:** (d_i) are matched patient differences; (t_{1-\alpha/2,n-1}) is a Student-t quantile; (\hat\theta^*) is a statistic from a bootstrap resample; (q_p) is its (p)-quantile.
5. **Purpose in the syllabus:** Show plausible parameter values and estimation uncertainty, complementing a binary significance decision.
6. **Application in SPIKE-ATTENTION:** The analyzer uses percentile bootstrap intervals for each model and paired patient bootstrap intervals for ST-SNN-minus-baseline differences. Paired samples are resampled together by patient. In the pilot, (B=2000) resamples and (n=9).
7. **Relevant source files:** `src/spikeattn/statistics/confidence_intervals.py`; `src/spikeattn/statistics/experiment_analyzer.py`; `src/spikeattn/statistics/ttest.py`.
8. **Advantages:** Communicates magnitude and uncertainty, including whether differences of practical interest remain plausible.
9. **Disadvantages and limitations:** With nine patients, intervals can be unstable and do not compensate for a biased or unrepresentative sample. A 95% interval is not a 95% probability statement about this already-computed fixed interval.
10. **Practical example using the project's ECG data:** The paired bootstrap interval for ST-SNN minus CNN balanced accuracy was ([-0.2433,0.0094]), showing that the pilot data remain compatible with a range from a substantial ST-SNN disadvantage to near no difference.

### 3.10 Predictive Analysis

1. **Terminology:** Training, validation, held-out test set, classification, generalization.
2. **Definition:** Learning a function from labeled examples to predict class labels or class probabilities for new examples.
3. **Mathematical formulation:** A classifier maps beat (x_i) to class probabilities (p_{ic}=P(Y_i=c\mid x_i;\theta)); prediction is (\hat y_i=\arg\max_c p_{ic}). For neural classifiers, the training objective includes cross-entropy (L=-\frac1N\sum_i\log p_{i,y_i}).
4. **Variable definitions:** (x_i) is an ECG beat, (Y_i) its true class, (c) one of N/S/V/F/Q, (\theta) learned model parameters, and (N) the number of training examples.
5. **Purpose in the syllabus:** Apply statistical and computational methods to forecast or classify future observations and evaluate generalization.
6. **Application in SPIKE-ATTENTION:** The runner fits five model families, uses validation data for checkpoint/selection decisions, and saves separate validation and test predictions/metrics. It computes accuracy, balanced accuracy, macro/weighted precision-recall-F1, per-class values, confusion matrices, and OVR probability metrics. Undefined probability metrics are reported for unsupported classes; Q had no test positives in this fold.
7. **Relevant source files:** `experiments/run_experiment.py`; `src/spikeattn/evaluation/model_evaluator.py`; `src/spikeattn/statistics/classification_metrics.py`; `configs/experiment.yaml`.
8. **Advantages:** Measures predictive performance on held-out patients and supports class-sensitive assessment under imbalance.
9. **Disadvantages and limitations:** A single fold/epoch is a noisy estimate. Overall accuracy can be misleading under class imbalance. Model comparisons require patient-level inference beyond beat-level metrics.
10. **Practical example using the project's ECG data:** A test ECG beat is assigned to one of N/S/V/F/Q. In the pilot, ST-SNN correctly classified 42.0% of all test beats and achieved balanced accuracy 0.4261; both are held-out fold metrics, not population-level proof.

## 4. Statistical Methods and Equations

This section collects the core formulas above and clarifies which were used. The test fold metrics are computed from beat predictions, while the primary inferential formulas use patient-level balanced accuracies.

| Method | Formula | Project meaning |
|---|---|---|
| Paired t-test | (t=\bar d/(s_d/\sqrt n),\ df=n-1) | Tests mean ST-SNN minus baseline balanced accuracy across the same patient groups. |
| One-way ANOVA | (F=(SS_B/(k-1))/(SS_W/(N-k))) | Implemented only; not part of pilot primary analysis. The repeated-measures alternative is relevant to matched model scores, but the project uses Friedman for its secondary matched omnibus test. |
| Chi-square | (\chi^2=\sum_{ij}(O_{ij}-E_{ij})^2/E_{ij}) | Implemented only; no pilot inferential chi-square result. |
| Pearson correlation | (r=\sum(x_i-\bar x)(y_i-\bar y)/\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}) | Association measure; exploratory, not primary. |
| Spearman correlation | (\rho_s=\operatorname{corr}(\operatorname{rank}(x),\operatorname{rank}(y))) | Rank association used by the exploratory relationship path when data are sufficient. |
| OLS regression | (y_i=\beta_0+\beta_1x_i+\epsilon_i), minimize (\sum_i(y_i-\hat y_i)^2) | Exploratory cost/performance relationship only. |
| Mean and sample standard deviation | (\bar x=\sum x_i/n;\ s=\sqrt{\sum(x_i-\bar x)^2/(n-1)}) | Describes patient score center and spread. |
| Confidence interval | (\bar d\pm t_{1-\alpha/2,n-1}s_d/\sqrt n) | Paired t interval; bootstrap intervals instead use empirical quantiles of resampled patient statistics. |
| Classification accuracy | (\sum_i\mathbf1(\hat y_i=y_i)/N) | Fraction of beats classified correctly. |
| Balanced accuracy | (\frac1C\sum_{c=1}^C TP_c/(TP_c+FN_c)) | Mean recall across the fixed class set; helps expose majority-class dominance. |

For the pilot’s paired comparisons, (d_i=BA_{ST,i}-BA_{baseline,i}). A negative mean difference indicates lower ST-SNN balanced accuracy. For Holm adjustment of four p-values, sort (p_{(1)}\le\cdots\le p_{(4)}), compare sequentially to (\alpha/(4-j+1)), and report adjusted p-values for the family. The project uses this to control family-wise error across ST-SNN versus four baselines.

The primary omnibus comparison is **Friedman**, a rank-based matched-samples test, not ANOVA. In the pilot it tests whether model ranks differ across nine common patient groups; its statistic was 2.4358 with four degrees of freedom. The code also reports Kendall’s (W) as a rank-based effect summary. This is noted here to explain the actual hypothesis-testing workflow, not to add it as a separate syllabus mapping row.

## 5. Practical Application in the Project

1. **ECG data preparation:** Read MIT-BIH records and annotations, apply the configured band-pass filter, segment beats, map annotations into N/S/V/F/Q, and retain patient group IDs.
2. **Model predictions:** Fit models on training groups. Validation groups guide neural checkpoint selection. The held-out test groups do not guide model selection.
3. **Metric calculation:** Compare predictions with labels. Save accuracy, balanced accuracy, macro and weighted precision/recall/F1, class support, confusion matrix, and probability metrics. Test and validation artifacts are distinct.
4. **Patient-level aggregation:** Calculate performance within each patient group. This yields a matched score for every model on the same patients.
5. **Hypothesis formulation:** For each baseline, (H_0) is zero mean difference in patient balanced accuracy; (H_1) is a two-sided nonzero difference.
6. **Assumption checking:** Examine paired differences using Shapiro-Wilk and an IQR outlier rule. These checks inform the test selector but cannot establish assumptions, especially at (n=9).
7. **Statistical test selection:** The primary analyzer selects paired t or Wilcoxon signed-rank based on its rule. It additionally runs a matched Friedman omnibus test across models.
8. **Multiple-comparison correction:** Holm adjusts the four primary ST-SNN-versus-baseline p-values. Conclusions use adjusted values for this family.
9. **Effect size:** The analyzer reports a paired standardized effect (Hedges-corrected (d_z)) when paired t is used, or rank-biserial effect for Wilcoxon. Friedman’s Kendall (W) describes omnibus rank effect.
10. **Confidence intervals and reporting:** Patient-level bootstrap intervals summarize each model and paired differences. Report sample size, raw and adjusted p-values, effects, intervals, predictive metrics, and limitations together.

Patient groups are the primary independent units because beats from one patient share physiology, acquisition conditions, and record-specific characteristics. Treating every beat as independent would inflate the apparent sample size and make uncertainty too small. Matched analysis also ensures each model is compared on the same people.

## 6. Actual Experimental Results

### 6.1 Scope and predictive results

The saved `results/infra_pilot_cuda` run is a **preliminary pilot**: seed 0, fold 0 only, one training epoch, five models, nine test patient groups, 24,698 test beats. Only one of five configured folds was run because the command specified `--max-folds 1`. The four other folds remain unexecuted in this run. The table below is held-out **test** performance; validation scores were used for model/checkpoint selection and are not mixed into this table.

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro precision | Macro recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| `svm` (calibrated linear SGD margin classifier) | 0.9115 | 0.3332 | 0.2823 | 0.3590 | 0.2666 | 0.8889 |
| `cnn` | 0.5712 | 0.4639 | 0.2358 | 0.2511 | 0.3711 | 0.6588 |
| `rnn` | 0.7702 | 0.4547 | 0.2813 | 0.2759 | 0.3637 | 0.8082 |
| `plain_snn` | 0.5658 | 0.4381 | 0.2374 | 0.2549 | 0.3504 | 0.6649 |
| `st_snn` | 0.4201 | 0.4261 | 0.1986 | 0.2460 | 0.3409 | 0.5303 |

The test support was N=22,199, S=695, V=1,769, F=35, and Q=0 (24,698 total). Therefore Q has no true test examples; Q-specific recall and one-vs-rest probability metrics are undefined. The saved metric implementation retains the fixed five labels and records undefined AUC classes explicitly. Accuracy is dominated by N prevalence; balanced accuracy and class-specific results are essential context. See `results/infra_pilot_cuda/metrics/test_metrics.csv`, `test_per_class.csv`, and `test_confusion_matrices.csv`.

### 6.2 Patient-level inferential results

The outcome used for primary comparisons is patient-level balanced accuracy. All comparisons below are paired and two-sided. All four pilot comparisons selected paired t-tests, with (n=9), (df=8), and Holm adjustment across the four tests.

| ST-SNN minus baseline | Mean difference | t (df=8) | Raw p | Holm p | Hedges g | Paired bootstrap 95% CI | Holm decision at .05 |
|---|---:|---:|---:|---:|---:|---:|---|
| `svm` | -0.0311 | -0.5504 | 0.59704 | 1.00000 | -0.1657 | [-0.1357, 0.0798] | Do not reject |
| `cnn` | -0.1187 | -1.7899 | 0.11125 | 0.33376 | -0.5389 | [-0.2433, 0.0094] | Do not reject |
| `rnn` | -0.1134 | -2.6979 | 0.02716 | 0.10864 | -0.8123 | [-0.1870, -0.0302] | Do not reject |
| `plain_snn` | 0.0132 | 0.3866 | 0.70916 | 1.00000 | 0.1164 | [-0.0420, 0.0847] | Do not reject |

The matched Friedman test across all five models returned statistic 2.4358, (df=4), (p=0.65618), (n=9), and Kendall’s (W=0.0677). Per-model patient-level percentile bootstrap 95% CIs were: ST-SNN 0.4103 [0.3565, 0.4636], SVM 0.4413 [0.3197, 0.5627], CNN 0.5290 [0.3854, 0.6742], RNN 0.5237 [0.4355, 0.6074], and plain SNN 0.3971 [0.3372, 0.4634]. The report specifies (B=2000) bootstrap replicates.

### 6.3 Efficiency interpretation

The output distinguishes metric kinds: parameter count and weight storage are **computed** from model parameters; process RSS, peak RSS, CUDA memory, and latency are **measured** by the profiler; CMOS-style energy is **estimated**, not physically measured energy. Inference latency uses five warm-up calls and 30 repetitions on the same 10 test examples, so its units describe a 10-beat batch. The SVM uses CPU; neural models use CUDA. Their timings are not a controlled same-device comparison. The project should not present estimated energy as measured energy.

## 7. Interpretation

The pilot provides no statistically supported evidence that ST-SNN is superior. ST-SNN test balanced accuracy is below CNN and RNN, and all four Holm-adjusted comparisons fail to reject the null. The Friedman result is also nonsignificant and Kendall’s (W) is small in this pilot. These findings do not establish equivalence: nine patients, one fold, and one epoch provide limited information.

The raw ST-SNN/RNN paired p-value is 0.02716, below 0.05, while its Holm-adjusted value is 0.10864. The raw p-value addresses that one comparison without accounting for the four-comparison family; the adjusted p-value is the relevant decision for the stated primary family. The generated report’s RNN line contains a raw-test interpretation saying “Reject H0” even though the adjusted decision is false. This document follows the adjusted decision and does not report a familywise rejection.

Statistical significance and practical significance are different. A small p-value does not show a useful clinical gain or acceptable computation cost; effect magnitude, confidence interval, class-wise performance, and deployment constraints also matter. Conversely, nonsignificance does not prove no difference. The pilot’s class Q absence further limits claims about five-class generalization. Efficiency data also mix CPU and GPU devices and include estimated energy; they cannot support a physically measured energy superiority claim.

## 8. Viva Preparation

1. **Why did you use a paired t-test?** Each model is evaluated on the same patient groups. The test analyzes within-patient score differences and controls some between-patient variability.
2. **What is the difference between paired and independent t-tests?** A paired test uses linked observations and tests their differences; an independent test assumes separate unrelated groups. The model scores are matched by patient.
3. **Why is the Friedman test relevant?** It is a nonparametric omnibus test for matched outcomes across several models. Patients are the repeated blocks. It is not ANOVA.
4. **What is the purpose of Holm correction?** It controls family-wise false-positive risk across the four primary ST-SNN-versus-baseline comparisons while being less conservative than simple Bonferroni in many cases.
5. **What is a p-value?** Under the null hypothesis and test assumptions, it is the probability of a result at least as extreme as the observed statistic. It is not the probability that the null is true.
6. **What is statistical significance?** A result meeting a preselected decision threshold after applying the designated correction. Here the relevant primary threshold is 0.05 after Holm adjustment.
7. **What is a confidence interval?** An interval produced by a procedure with a stated long-run coverage rate. It communicates plausible parameter values and estimation uncertainty.
8. **What is the difference between accuracy and balanced accuracy?** Accuracy weights each beat equally and is affected by class prevalence. Balanced accuracy averages recall across classes, giving each class equal weight.
9. **Why use patient-level comparisons?** Beats from the same patient are correlated. Patient-level analysis avoids treating clustered beats as independent replicates and compares models on matched subjects.
10. **What happens when normality of differences is not supported?** The analyzer’s rule selects Wilcoxon signed-rank rather than the paired t-test. With nine observations, the diagnostic has limited power and cannot prove normality.
11. **Why is correlation not causation?** Correlation measures co-variation; confounding factors such as model family, hardware, and implementation can explain an association.
12. **What does regression tell us here?** The implemented exploratory OLS path can describe a relationship such as latency versus performance. It is not the primary model-superiority test and does not establish a causal effect.
13. **Why is the current experiment inconclusive?** It used one of five folds, one seed, one epoch, and nine test patients; class Q was absent, and adjusted comparisons did not reject their null hypotheses.
14. **Why can raw and adjusted p-values differ?** Holm accounts for testing a family of four comparisons. A raw p-value may be below 0.05 while the familywise-adjusted p-value is above it, as with ST-SNN versus RNN.
15. **Does a nonsignificant result prove the models are equivalent?** No. Failure to reject is not evidence of equivalence; a predefined equivalence margin and an equivalence test would be needed.
16. **Why is the SVM result’s accuracy high while balanced accuracy is low?** The test fold is heavily imbalanced toward N. Predicting the majority class well can yield high accuracy while recall averaged across classes remains low.
17. **Why is class Q not interpreted in the pilot?** There are no Q-labeled true examples in this held-out fold, so its recall and class-specific ROC/PR metrics are undefined.
18. **Was the pilot’s energy physically measured?** No. Energy is labeled estimated. Latency and memory are profiled measurements; parameter count and weight memory are computed quantities.
19. **Was ANOVA used to compare the pilot models?** No. ANOVA utilities exist, but the primary analysis used paired comparisons and a matched Friedman test.
20. **What should happen before claiming model superiority?** Run the intended full fold/seed plan, verify every split and class support, preserve the patient as the inference unit, and report corrected tests, effect sizes, intervals, and limitations.

## Source and result references

- Experiment configuration: `configs/experiment.yaml`, `configs/dataset.yaml`, `configs/label_mapping.yaml`, `configs/analysis.yaml`, `configs/efficiency.yaml`.
- Experiment orchestration and evaluation: `experiments/run_experiment.py`, `src/spikeattn/evaluation/model_evaluator.py`, `src/spikeattn/statistics/classification_metrics.py`.
- Statistical implementation: `src/spikeattn/statistics/experiment_analyzer.py`, `ttest.py`, `test_selector.py`, `anova.py`, `chi_square.py`, `correlation.py`, `regression.py`, `descriptive_statistics.py`, `confidence_intervals.py`, `nonparametric_tests.py`, `multiple_comparison.py`, and `effect_size.py`.
- Pilot predictive metrics: `results/infra_pilot_cuda/metrics/test_metrics.csv`, `test_per_class.csv`, `test_confusion_matrices.csv`, and `per_patient.csv`.
- Pilot inference and intervals: `results/infra_pilot_cuda/report.md`.
- Pilot profiling: `results/infra_pilot_cuda/efficiency/measurements.csv`.
- Broader project explanation and methodological caveats: `report.md`.
