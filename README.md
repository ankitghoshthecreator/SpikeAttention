# SPIKE-ATTENTION

### A Hybrid Spatio-Temporal Attention Spiking Neural Network for Two-Channel ECG Analysis

**Academic subject:** Inferential Statistics and Predictive Analysis (ISPA)

> **Project status: design phase.** The documentation and architecture are complete; the software, experiments and results are **planned, not yet implemented**. Sections marked *(planned)* describe intended functionality. This README contains no experimental results, and none should be inferred.

---

## 1. Description

SPIKE-ATTENTION investigates whether a Spiking Neural Network (SNN) augmented with temporal attention and inter-lead attention predicts ECG beat classes on **unseen patients** better than conventional baselines, and whether any difference is **statistically reliable, practically meaningful and computationally worthwhile**.

It is an **ISPA project first**: deep learning supplies the predictive models; the core deliverable is a set of object-oriented Python modules for descriptive statistics, hypothesis testing, confidence intervals, effect sizes, correlation, regression, ROC analysis, multiple-comparison correction, assumption testing and efficiency analysis, coordinated by one `ExperimentAnalyzer`.

## 2. Problem Statement

ECG beats carry temporal and cross-lead patterns that distinguish normal from arrhythmic beats. Published comparisons often (a) leak data between training and test (beat-level or record-level splits), (b) treat thousands of correlated beats as independent observations, (c) report single numbers without uncertainty, and (d) assert efficiency gains for SNNs without measurement. This project designs an experiment that avoids those problems.

## 3. Research Objectives

1. Build a patient-wise, leakage-free ECG classification pipeline on MIT-BIH.
2. Compare ST-SNN with SVM, CNN, RNN and a plain SNN under one protocol.
3. Test performance differences with the correct statistical unit (patient), paired design, assumption-driven test selection, effect sizes, confidence intervals and multiple-comparison correction.
4. Quantify uncertainty of every headline estimate.
5. Measure computational cost and clearly separate **measured** from **estimated** quantities (energy is only ever estimated here, with a transparent method).
6. Report negative or inconclusive outcomes honestly.

## 4. Main Features *(planned)*

- Patient-grouped, class-stratified, repeated cross-validation; out-of-fold probabilities stored for all models.
- Delta (ON/OFF) spike encoding; LIF spiking convolution; surrogate-gradient training.
- Temporal attention and inter-lead attention (hybrid, real-valued attention on spike-derived features).
- Five models behind one `BaseModel` interface.
- 20+ statistical classes connected by `ExperimentAnalyzer`.
- Generated CSV, JSON, PNG and Markdown outputs in a reproducible `results/<run_id>/` directory.

## 5. ISPA Concepts Implemented *(planned)*

| ISPA Concept | Python Module | Class | Application in Project | Output |
|---|---|---|---|---|
| Descriptive statistics | `statistics/descriptive_statistics.py` | `DescriptiveStatistics` | Class distribution, ECG feature distributions, metric summaries (mean, median, mode, variance, SD, quartiles, IQR, range) | `descriptive_*.csv`, histograms, box/violin plots |
| Hypothesis testing framework | `statistics/hypothesis_testing.py`, `statistics/base.py` | `HypothesisTester`, `BaseStatisticalTest`, `TestResult` | H0/H1, alpha, critical region, one/two-tailed, Type I/II error, power | `hypothesis_results.json/csv` |
| t-tests | `statistics/ttest.py` | `OneSampleTTest`, `IndependentTTest`, `WelchTTest`, `PairedTTest`, `CorrectedResampledTTest` | Paired model comparison; group comparisons; fold-level corrected test | test result records, p-value chart |
| Chi-square | `statistics/chi_square.py` | `ChiSquareTest` | Split-distribution check; correctness-by-class association | contingency tables, results |
| ANOVA | `statistics/anova.py` | `OneWayANOVA`, `RepeatedMeasuresANOVA` | Feature-by-class; five-model comparison | ANOVA tables |
| Non-parametric tests | `statistics/nonparametric_tests.py` | `WilcoxonSignedRankTest`, `FriedmanTest`, `KruskalWallisTest` | Robust paired and multi-model comparison | results + post-hoc table |
| McNemar | `statistics/mcnemar_test.py` | `McNemarTest` | Paired pooled error comparison (exploratory) | 2Ã—2 discordance tables |
| Confidence intervals | `statistics/confidence_intervals.py` | `ConfidenceIntervalEstimator` | Wilson, normal, patient-clustered bootstrap | CI tables, CI plots |
| Effect size | `statistics/effect_size.py` | `EffectSizeAnalyzer` | Cohen's d, paired d, Hedges' g, rank-biserial, Kendall's W | effect-size table/plot |
| Correlation | `statistics/correlation.py` | `CorrelationAnalyzer` | Pearson/Spearman, matrices, significance | heatmaps, tables |
| Regression | `statistics/regression.py` | `RegressionAnalyzer` | Simple/multiple OLS, diagnostics, cost-vs-performance | coefficient tables, regression/residual plots |
| Predictive pipeline | `evaluation/cross_validation.py`, `prediction/predictor.py`, `evaluation/model_evaluator.py`, `models/baselines/*`, `training/hpo.py` | `PatientGroupCrossValidator`, `Predictor`, `ModelEvaluator`, `HyperparameterTuner` | Patient-wise CV, baselines, tuning, probability analysis | OOF prediction files, metric tables |
| Confusion matrix / metrics | `statistics/classification_metrics.py` | `ClassificationMetrics` | TP/TN/FP/FN, accuracy, precision, recall, specificity, F1, macro/weighted F1 | report tables, confusion-matrix PNG |
| ROC analysis | `statistics/roc.py` | `ROCAnalyzer` | One-vs-rest ROC Curve, TPR/FPR, thresholds, AU (Area Under the ROC Curve) | ROC PNG, threshold CSV |
| Multiple comparison | `statistics/multiple_comparison.py` | `MultipleComparisonCorrector` | Bonferroni, Holm, Benjaminiâ€“Hochberg over pre-defined families | adjusted p-value tables |
| Assumption testing | `statistics/assumption_tests.py` | `AssumptionTester` | Shapiroâ€“Wilk, Levene, outliers, Q-Q, sphericity | assumption report, Q-Q plots |
| Computational efficiency | `statistics/efficiency_analysis.py`, `evaluation/efficiency_profiler.py` | `EfficiencyAnalyzer`, `EfficiencyProfiler` | Latency, parameters, memory, spike activity, estimated energy | efficiency tables/plots |
| Workflow integration | `statistics/experiment_analyzer.py`, `statistics/test_selector.py` | `ExperimentAnalyzer`, `TestSelector` | Orchestrates all analyses on stored results | final JSON, Markdown report |

Techniques deliberately **not** used as primary analyses: t-tests on raw beat-level scores (pseudoreplication), independent t-tests on model scores from the same patients (the design is paired), time-series tests on single recordings (autocorrelated segments), mixed-effects and Bayesian models (optional future work).

## 6. Proposed Architecture

The system has a **predictive subsystem** (data â†’ preprocessing â†’ spike encoding â†’ models â†’ cross-validated predictions) and an **inferential subsystem** (`ExperimentAnalyzer` â†’ assumptions â†’ tests â†’ intervals â†’ effect sizes â†’ corrections â†’ interpretation â†’ plots â†’ report). They communicate only through stored artifacts, so statistics always come from real experimental output.

- Full design: [`arch.md`](arch.md)
- Requirements: [`prd.md`](prd.md)
- Diagram (GitHub-renderable Mermaid): [`arch.mmd`](arch.mmd)

## 7. Model Explanation

**Pipeline:** two-channel ECG â†’ bandpass filter â†’ normalization â†’ beat segmentation â†’ spike encoding â†’ spiking convolution (LIF) â†’ temporal attention â†’ inter-lead attention â†’ global pooling â†’ fully connected layer â†’ softmax â†’ predicted class.

**LIF neuron:**

```text
tau * dV(t)/dt = -(V(t) - V_reset) + R * I(t)
V(t) > V_threshold  ->  spike;   then  V(t) <- V_reset
```

Discretized with an Euler step; gradients flow through the non-differentiable spike via an arctan surrogate gradient.

**Spike encoding (default):** delta/threshold modulation. A lead emits an ON spike when the signal rises by a step Î¸ above its running reference and an OFF spike when it falls by Î¸. Two leads Ã— ON/OFF give four spike channels. Î¸ is derived from training data only and logged. *Limitation:* sensitive to Î¸ and noise.

**Attention:**

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Spikes are first converted into real-valued features (a short exponential trace). *Temporal attention* uses time steps as tokens; *inter-lead attention* uses the two leads as tokens. Because softmax attention is dense and real-valued, the network is a **hybrid**, not fully event-driven, which limits any efficiency claim.

**Baselines:** SVM (RBF on handcrafted features), 1-D CNN, GRU-based RNN, and a plain SNN (ST-SNN minus attention), with matched parameter budgets for the neural models and equal tuning budgets.

**Training:** class-weighted cross-entropy, Adam, early stopping on validation patients, fixed seeds.

## 8. Dataset Information

**MIT-BIH Arrhythmia Database**, PhysioNet: https://physionet.org/content/mitdb/

Facts to be verified and recorded by the pipeline from the downloaded data: 48 half-hour two-channel recordings from 47 subjects (records 201 and 202 are the same subject), 360 Hz, beat-level annotations; records 102, 104, and 114 lack the usual MLII first lead (records 102 and 104 have V5/V2, while record 114 has V5/MLII); records 102, 104, 107 and 217 contain paced beats (excluded by default, configurable). Dataset statistics (beat counts per class, per-patient class presence) are **computed from the data at run time and never typed into code or documentation**.

**Label mapping:** beat annotation symbols are mapped to AAMI EC57 classes (N, S, V, F, Q) using the explicit table `configs/label_mapping.yaml`. These are *beat-type* labels, **not clinical diagnoses**.

**Record-level vs patient-level splitting:** record-level splitting treats recordings as independent and can leak when a subject has more than one record; patient-level splitting guarantees no subject appears in more than one of train/validation/test. This project uses patient-level splitting only, with automated disjointness assertions.

**Limitations:** coarse AAMI grouping; very rare classes (S, F, Q) with few patients; annotation uncertainty; small, historical single-source population (limited external validity); class imbalance.

## 9. Statistical Design in Brief

| Level | Role |
|---|---|
| Beat/segment | Model input only; **never** an independent observation |
| Patient group | **Primary statistical unit** (per-patient balanced accuracy, averaged over seeds, paired across models) |
| Seed | Secondary unit (training-randomness analysis) |
| CV fold | Only via the Nadeauâ€“Bengio corrected resampled t-test |

- Primary family: ST-SNN vs each of four baselines, paired on patients, Holm-corrected.
- Test choice (paired t vs Wilcoxon) follows a pre-specified assumption-based rule; both are stored.
- Effect size and confidence interval accompany every p-value; the smallest effect size of interest (default 0.02 balanced accuracy, configurable) is declared before experiments.
- McNemar, chi-square, one-way ANOVA, correlation and regression are labelled **exploratory** unless justified otherwise.
- A non-significant normality test is never reported as evidence of normality.

## 10. Technology Stack

Python 3.11 Â· NumPy Â· Pandas Â· SciPy Â· statsmodels Â· scikit-learn Â· PyTorch (custom LIF/surrogate gradient; SpikingJelly optional cross-check) Â· WFDB Â· Matplotlib Â· Seaborn Â· PyYAML Â· pytest.

## 11. Folder Structure

```text
spike-attention/
â”œâ”€â”€ configs/                 # dataset, label mapping, models, experiment, analysis, efficiency
â”œâ”€â”€ data/{raw,processed,metadata}/
â”œâ”€â”€ src/spikeattn/
â”‚   â”œâ”€â”€ data/  preprocessing/  encoding/  models/{snn,baselines}/
â”‚   â”œâ”€â”€ training/  prediction/  evaluation/
â”‚   â”œâ”€â”€ statistics/          # all ISPA modules + experiment_analyzer.py
â”‚   â”œâ”€â”€ visualization/  reporting/  utils/
â”œâ”€â”€ experiments/  results/  reports/  tests/
â”œâ”€â”€ main.py  requirements.txt  .env.example  .gitignore
â””â”€â”€ prd.md  arch.md  arch.mmd  README.md
```

(The package is named `spikeattn` so the `statistics` subpackage cannot shadow Python's standard library.)

## 12. Installation and Environment *(planned; commands become valid after implementation milestone M0)*

```bash
git clone <repository-url> spike-attention && cd spike-attention
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # set DATA_DIR, RESULTS_DIR, DEVICE (cpu|cuda)
```

CPU is sufficient for all statistics; a GPU is optional for training deep models.

## 13. Dataset Preparation *(planned)*

```bash
python main.py data download      --config configs/dataset.yaml
python main.py data catalog       --config configs/dataset.yaml
python main.py data preprocess    --config configs/dataset.yaml
```

## 14. Running Experiments *(planned)*

```bash
python main.py experiment run --config configs/experiment.yaml          # all models, folds, seeds
python main.py experiment run --config configs/experiment.yaml --models st_snn plain_snn --seeds 0 1
python main.py experiment status --run-id <run_id>
```

Runs are resumable and write to `results/<run_id>/` with a `manifest.json` (configs, hash, seeds, package versions, hardware).

## 15. Statistical Analysis *(planned)*

```bash
python main.py analyze --run-id <run_id> --config configs/analysis.yaml
python main.py report  --run-id <run_id>
```

`analyze` runs the full `ExperimentAnalyzer` workflow: descriptive statistics â†’ assumption checks â†’ test selection â†’ hypothesis tests â†’ confidence intervals â†’ effect sizes â†’ multiple-comparison correction â†’ correlation/regression â†’ ROC analysis â†’ efficiency analysis â†’ interpretation. `report` produces the Markdown report and figures.

## 16. Expected Output Structure

```text
results/<run_id>/
â”œâ”€â”€ manifest.json
â”œâ”€â”€ splits/        predictions/   tuning/   metrics/   efficiency/
â”œâ”€â”€ statistics/    # JSON/CSV for every test, CI, effect size, correction
â”œâ”€â”€ figures/       # PNG
â”œâ”€â”€ reports/report.md
â””â”€â”€ logs/run.log
```

**Planned figures:** histograms, box plots, violin plots, confusion matrices, ROC Curves, correlation heatmaps, regression plots, confidence interval plots, model comparison charts, effect-size comparison, p-value comparison, efficiency comparison. All are generated from stored experimental outputs; none are illustrative.

## 17. Testing *(planned)*

```bash
pytest -q                       # all tests
pytest tests/unit/statistics -q # statistical classes vs SciPy/statsmodels references
pytest tests/unit/data -q       # leakage and label-mapping tests
pytest tests/integration -q     # synthetic-fixture end-to-end analysis
```

Test fixtures are synthetic and used only to verify code; they are never reported as findings.

## 18. Research Integrity Statement

No p-values, confidence intervals or performance numbers are hard-coded. No superiority claim is made before experimentation. Energy is never reported as measured unless directly measured; otherwise it is labelled an estimate with its constants and assumptions. Inconclusive and negative results will be reported in the same format as positive ones.

## 19. Limitations

Small patient population (limited power); coarse beat-level labels; hybrid attention is not event-driven; delta encoding is parameter-sensitive; the efficiency analysis is hardware-specific; single-database evidence; conclusions apply to MIT-BIH-like data only and carry no clinical claim.

## 20. Future Improvements

External validation (additional ECG databases); mixed-effects models for clustered data; spike-driven (softmax-free) attention; alternative encoders; direct power measurement or neuromorphic deployment; calibration analysis expansion.

## 21. References

- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45â€“50.
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215â€“e220.
- ANSI/AAMI EC57: Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.
- de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Trans. Biomed. Eng.*, 51(7), 1196â€“1206.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51â€“63.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52, 239â€“281.
- DemÅ¡ar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1â€“30.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895â€“1923.
- Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Statistics*, 6, 65â€“70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1), 289â€“300.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209â€“212.
- Horowitz, M. (2014). Computing's energy problem. *ISSCC* (source of commonly cited per-operation energy figures; used only as configurable, assumption-labelled constants).

*Reference details should be verified when the final report is written.*
