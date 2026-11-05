# SPIKE-ATTENTION: Project and Research Guide

This document explains the runnable SPIKE-ATTENTION project from the research question through data preparation, model training, inference, ISPA analysis, and the generated results. It is written as a project handover: a reader should be able to use it to explain what the project does, why each stage exists, where the implementation lives, and what the current experiment can and cannot show.

> **Scope note:** this guide describes the code currently in the repository. Some older design documents, particularly parts of `README.md`, `prd.md`, and `arch.md`, describe planned behavior or older model details. Where those differ from the runnable code, this guide names the implemented behavior.

## 1. The project in one paragraph

SPIKE-ATTENTION is a research pipeline for classifying individual heartbeats from two-channel ECG recordings into five AAMI beat classes: **N** (normal and related), **S** (supraventricular ectopic), **V** (ventricular ectopic), **F** (fusion), and **Q** (unknown or paced categories as defined by the mapping file). It compares a spatio-temporal attention spiking neural network (ST-SNN) with four baselines: a calibrated linear margin classifier called `svm`, a 1D CNN, a bidirectional GRU RNN, and a plain SNN. The research question is whether ST-SNN performs better on **patients withheld from model fitting**, and whether any observed difference is reliable and useful after accounting for patient-to-patient variation, multiple comparisons, and computational cost. Beats are the model inputs; patients are the independent units for primary statistical inference.

The work has two connected parts:

1. **Predictive pipeline:** transform ECG records into labeled beats, make patient-disjoint train/validation/test partitions, fit models, and save beat-level probabilities and labels.
2. **ISPA pipeline:** calculate performance and efficiency summaries, compare models with matched patient-level statistics, quantify uncertainty and effect sizes, control multiple testing, and produce a Markdown report.

## 2. The sequence from data to conclusion

```mermaid
flowchart LR
    A[MIT-BIH records and annotations] --> B[Catalog patients and beat labels]
    B --> C[Filter signals and cut 360-sample beats]
    C --> D[Create patient-disjoint five-fold splits]
    D --> E[Fit model on training patients]
    E --> F[Use validation patients for early stopping]
    F --> G[Predict validation and held-out test beats]
    G --> H[Save probabilities, predictions, metrics and plots]
    H --> I[Aggregate test balanced accuracy by patient]
    I --> J[Paired tests, Friedman, Holm and bootstrap]
    J --> K[Report findings, uncertainty and limitations]
```

### Stage 1 — Define the dataset and labels

The dataset configuration names the MIT-BIH Arrhythmia Database records, their annotation source, the sampling rate, beat window size, filter band, and the paced-record exclusion policy. The label mapping configuration translates annotation symbols into the project’s fixed five-class order. Records `201` and `202` belong to the same patient group; the data catalog maps them to one statistical group so that one subject cannot occur on both sides of a split.

The pipeline does not treat beats as independent split units. A patient can contribute many beats, and those beats share subject-specific characteristics. Splitting beats randomly would allow the model to see one patient during training and be tested on that same patient, inflating estimates of performance on unseen patients.

### Stage 2 — Filter and segment

`BandpassFilter` applies a Butterworth band-pass filter to the two recorded channels. `BeatSegmenter` uses each annotation sample as the beat center and extracts a fixed 360-sample segment. At 360 Hz, this is one second of signal per beat. Unsupported annotation symbols follow the configured omit-and-count policy. The processed cache stores signals, integer labels, group IDs, beat IDs, and R-peak indices; metadata is stored in a matching CSV.

Filtering is deterministic and has no learned parameters. A training-only `LeadNormalizer` is implemented, but the current `run_experiment.py` path does not call it. The SVM baseline fits its feature scaler on its training partition. The SNN delta encoder obtains its threshold from the first training batch seen by the model, rather than fitting it across the complete training partition; this is a reproducibility and methodology limitation to address in future work.

### Stage 3 — Make grouped folds and nested validation partitions

`PatientGroupCrossValidator` uses `StratifiedGroupKFold` at the patient-group level to create the outer folds. Stratification uses each patient’s dominant beat class, so rare class distributions are only approximately balanced. Within each outer training portion, `PatientSplitter.inner_split` reserves validation patient groups (configured as 20%). The test groups are untouched while a model trains and while early stopping chooses its checkpoint. Disjointness checks ensure train, validation, and test group sets do not overlap. Each split JSON records the group assignments and data hash.

The default configuration contains five folds and five seeds (`0`–`4`). A **five-fold benchmark for one seed** executes five outer folds; using all five configured seeds executes 25 folds per model. `--max-folds 1` deliberately limits a pilot to fold 0.

### Stage 4 — Fit and select models

The training runner creates a fresh model for each model/seed/fold combination. The four neural models are placed on the configured compute device; the `svm` estimator uses scikit-learn and stays on CPU. Neural-network early stopping monitors validation balanced accuracy and restores the best validation checkpoint. The test partition is not used for early stopping, hyperparameter selection, or model selection.

The trainer uses AdamW, class-weighted cross-entropy, gradient clipping, a cosine learning-rate schedule, and a fixed epoch ceiling. The command-line `--epochs` value overrides the configured/default maximum. A one-epoch pilot tests the pipeline; it is not a converged model comparison.

### Stage 5 — Predict and save

`Predictor` produces a probability vector in the fixed class order for each beat. The saved prediction row contains beat ID, patient group, record, seed, fold, model, true label, predicted label, and five class probabilities. Validation predictions and test predictions are written into separate directories. Test metrics are calculated only after fitting is complete.

The evaluator produces beat-level fold metrics, per-class metrics, support, confusion matrices, OVR ROC-AUC and average precision, plus per-patient balanced accuracy/accuracy/F1. Undefined ROC/PR values are marked if a class has no positive or no negative cases. Classification metrics use fixed five-class labels and `zero_division=0`; support and status fields make absent classes visible.

### Stage 6 — Infer at the patient level

For the primary model comparisons, a patient’s test beats are first reduced to one balanced-accuracy value per patient and model. Models are paired on the same patient IDs. The analyzer then selects a paired t-test or Wilcoxon signed-rank test according to the configured assumption rule, runs a matched Friedman omnibus test across models, applies Holm correction to the four planned ST-SNN comparisons, estimates paired effect sizes, and bootstraps patient-level intervals. The four pairwise comparisons are exploratory and two-sided in the current implementation.

This avoids the main form of pseudoreplication: pretending that thousands of correlated beats are thousands of independent statistical observations. A patient is the primary observation for the model-level inference; a beat is an observation only for descriptive classification counts and confusion matrices.

## 3. What “ISPA” means here

ISPA is **Inferential Statistics and Predictive Analysis**. Predictive models produce labels and probabilities; inferential statistics assess how much confidence to place in comparisons between models. The project’s intended logic is:

1. Report descriptive performance before making claims.
2. Use the design’s matched patient groups to compare models.
3. Check assumptions on paired differences before choosing a parametric or rank-based test.
4. Report uncertainty and effect magnitude, not only a p-value.
5. Correct the planned family of four ST-SNN comparisons for multiplicity.
6. Treat model/beat efficiency measures according to whether they were measured, computed, or estimated.
7. Limit the conclusion to the evaluated patients, folds, seeds, training procedure, and dataset.

### ISPA methods available in the code

| Concept | Main module | Role in this project |
|---|---|---|
| Descriptive statistics | `statistics/descriptive_statistics.py` | Summarizes patient-level scores (mean, median, spread, quartiles, range). |
| Hypothesis representation | `statistics/base.py`, `statistics/hypothesis_testing.py` | Carries H0/H1, alpha, tail, unit, validity, and result metadata. |
| Paired t-test | `statistics/ttest.py` | Tests whether the mean within-patient score difference differs from zero. |
| Assumption checks and test selection | `statistics/assumption_tests.py`, `statistics/test_selector.py` | Shapiro-Wilk on paired differences and extreme-outlier checks choose paired t or Wilcoxon. A non-significant Shapiro result is not proof of normality. |
| Wilcoxon signed-rank | `statistics/nonparametric_tests.py` | Rank-based paired alternative when t-test assumptions are doubtful. |
| Friedman test | `statistics/nonparametric_tests.py` | Matched omnibus comparison across three or more models on the same patients. |
| Holm correction | `statistics/multiple_comparison.py` | Controls family-wise error across the four ST-SNN-versus-baseline comparisons. |
| Effect sizes | `statistics/effect_size.py` | Paired Hedges’ g for t-test results, rank-biserial correlation for Wilcoxon, Kendall’s W for Friedman. |
| Confidence intervals | `statistics/confidence_intervals.py` | Wilson and normal intervals are available; the main report uses percentile bootstrap intervals resampling patient-level scores and paired score differences. |
| Classification metrics | `statistics/classification_metrics.py` | Accuracy, balanced accuracy, precision, recall, specificity, F1, and confusion-matrix decomposition. |
| ROC analysis | `statistics/roc.py`, `evaluation/model_evaluator.py` | OVR probability curves and macro ROC/PR summaries; absent classes are marked undefined for probability metrics. |
| Other tests | `statistics/anova.py`, `chi_square.py`, `mcnemar_test.py` | Reusable exploratory test implementations; they are not the primary inferential path in the current `ExperimentAnalyzer` report. |
| Relationships and regression | `statistics/correlation.py`, `statistics/regression.py` | Correlates model-level efficiency with performance and fits exploratory OLS models. With only five models, these relationships are descriptive/exploratory. |
| Efficiency accounting | `evaluation/efficiency_profiler.py`, `statistics/efficiency_analysis.py` | Measures latency/RSS/GPU memory, computes parameter/weight storage, and labels CMOS energy as estimated. |

### The t-test example: `src/spikeattn/statistics/ttest.py`

`PairedTTest` receives either one vector of differences or two equally sized vectors (one per model). For patient `i`, the difference is:

```text
d_i = balanced_accuracy(ST-SNN, patient_i) - balanced_accuracy(baseline, patient_i)
```

The test computes the mean difference `d̄`, sample standard deviation `s_d`, standard error `s_d / sqrt(n)`, and statistic `t = d̄ / (s_d / sqrt(n))` with `n - 1` degrees of freedom. A two-sided p-value asks whether a difference in either direction is compatible with a zero-mean paired difference under the model assumptions. It also returns a two-sided mean-difference interval and paired standardized effect (Hedges’ g correction applied to paired `d_z`).

Pairing matters because the same patients are evaluated by both models. It removes some between-patient variation from the comparison. It does not turn nine patients into a large study; a small `n` still gives imprecise estimates and limited power. The analyzer chooses this test only when its rule finds no evidence against normality in differences and no extreme outliers; otherwise it uses Wilcoxon. The rule is algorithmic, not a guarantee that assumptions are true.

### Interpreting p-values, effects, and intervals

- The **raw p-value** is for one comparison before multiplicity correction.
- The **Holm-adjusted p-value** is the decision statistic for the planned family of four tests.
- **Hedges’ g** or **rank-biserial correlation** describes effect magnitude in a paired design; it is not a substitute for uncertainty.
- The **paired bootstrap CI** resamples patient-level paired differences, keeping each patient’s two model scores together.
- A p-value above alpha does not prove models are equivalent. Equivalence would require a predeclared margin and an appropriate equivalence test.
- Statistical significance does not automatically imply clinical usefulness or a computational advantage.

## 4. Models and the signals they see

### Baselines

- **`svm`**: the runnable code extracts moments, extrema, energy, inter-lead correlation, and sub-window energy features, standardizes them, then fits an `SGDClassifier` with modified-Huber loss and probability calibration. Although the class is named `SVMBaseline` and older docs mention an RBF SVM, the implementation is a linear SGD margin classifier; results should be described that way until the baseline is changed or the documentation reconciled. scikit-learn executes it on CPU.
- **`cnn`**: three 1D convolutional stages with batch normalization, ReLU, and pooling, followed by global pooling and a five-class linear head. It processes the two ECG channels directly.
- **`rnn`**: two-layer bidirectional GRU over time, followed by temporal mean pooling and a five-class linear head. It processes a 360-step sequence and is computationally expensive on CPU; it runs on CUDA when configured and available.
- **`plain_snn`**: delta-spike encoding, two spiking convolution/LIF blocks, global temporal pooling, and a class head. It is the SNN control without attention.

### Proposed model: `st_snn`

The ST-SNN is a hybrid spiking model:

1. **Delta encoder:** two continuous leads become four event channels: ON/OFF for lead 0 and ON/OFF for lead 1.
2. **Spiking convolution block 1:** temporal convolution, batch normalization, and LIF dynamics extract local event features.
3. **Inter-lead attention:** a learned 2×2 attention map lets one lead’s representation influence the other.
4. **Spiking convolution block 2:** further temporal feature extraction.
5. **Temporal self-attention:** each time token can attend to other time tokens.
6. **Global temporal pooling and linear classifier:** produces five logits and softmax probabilities.

LIF means **Leaky Integrate-and-Fire**. A neuron integrates input current into a membrane voltage, leaks toward reset, emits a spike when voltage crosses threshold, then resets. A hard threshold has no ordinary derivative, so a smooth arctangent surrogate is used during backpropagation. Attention uses real-valued dense operations on spike-derived features; the network is therefore hybrid, not fully event-driven. It should not be assumed energy-efficient merely because it contains spiking neurons.

The current delta encoder has a notable implementation detail: in `forward`, an unfitted encoder fits itself on the first batch it sees. That is a training batch, so validation/test labels or signals do not fit the threshold, but the threshold is based on only that first batch rather than a deterministic full-training summary. `encode()` also copies signals to CPU and performs sequential threshold loops there before returning spikes to the model device. This is one reason GPU training does not make every SNN operation GPU-native.

## 5. Current research outcome: completed pilot

The current pilot is `results/infra_pilot_cuda/`. It evaluated all five models on seed 0, fold 0, for one training epoch. The execution log reports `device=cuda`; efficiency output confirms CUDA for CNN, RNN, plain SNN, and ST-SNN. SVM is CPU-based by design of its scikit-learn implementation. The fold status file marks all five models completed.

### Held-out test performance from this pilot

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC OVR | Macro PR-AUC |
|---|---:|---:|---:|---:|---:|
| SVM-like SGD margin model | 0.9115 | 0.3332 | 0.2823 | 0.5676 | 0.3535 |
| CNN | 0.5712 | 0.4639 | 0.2358 | 0.6651 | 0.3516 |
| Bi-GRU RNN | 0.7702 | 0.4547 | 0.2813 | 0.7271 | 0.4043 |
| Plain SNN | 0.5658 | 0.4381 | 0.2374 | 0.7313 | 0.3997 |
| ST-SNN | 0.4201 | 0.4261 | 0.1986 | 0.6532 | 0.3936 |

The test fold contains 24,698 beats from nine patient groups. Class Q has zero support in this fold, so its per-class discrimination and ROC/PR metrics are undefined; the macro probability metrics average the classes for which both positive and negative examples exist. Macro classification metrics use the fixed five-class label set and the declared zero-division policy. See `metrics/test_per_class.csv` and the confusion matrices before interpreting aggregate scores.

The key research conclusion is **inconclusive, with no evidence that ST-SNN is superior**. On this one-epoch fold its balanced accuracy (0.4261) is below CNN (0.4639) and RNN (0.4547). Patient-level analysis uses nine matched groups. The Friedman omnibus test is not significant (`p = 0.656`, Kendall’s `W = 0.068`). Of the four ST-SNN pairwise comparisons, RNN has raw `p = 0.027`, but Holm-adjusted `p = 0.109`; the adjusted test does not reject the null. No other adjusted comparison rejects. The generated report’s interpretation string for that RNN row says “Reject H0” based on the unadjusted test result; the adjacent Holm decision is `False`. For the research conclusion, use the Holm-adjusted decision, not that raw-result wording.

This pilot is a smoke test, not a final benchmark: one seed, one of five folds, one epoch, nine test patients, and a rare class absent from the held-out fold. It neither establishes superiority nor equivalence. The intended final result requires all five folds and should state the seed coverage explicitly. Five folds from seed 0 are not the same as five folds across all five configured seeds.

### Efficiency interpretation

The profiler times repeated calls on the same first 10 test examples. The latency values are therefore milliseconds per 10-example inference batch, not single-beat latency, even though an older report heading says “per beat.” Process RSS includes the running Python process and loaded libraries; peak process RSS is sampled and may miss short spikes. GPU peak memory records allocated CUDA memory. Parameter counts and weight storage are computed. Horowitz-style energy is estimated, not physically measured. CPU SVM latency and CUDA neural-model latency should not be treated as a controlled same-hardware comparison.

## 6. How to run the project

The current environment uses Python 3.11 and CUDA-enabled PyTorch 2.6.0+cu124; the configured device is `cuda`. The project runner fails rather than silently falling back to CPU if CUDA is unavailable. The current machine has an NVIDIA GeForce RTX 3050 Laptop GPU; SVM remains CPU-based.

Typical commands from the repository root:

```powershell
# Data ingestion and preparation (use only if the corresponding artifacts need rebuilding)
.\.venv\Scripts\python.exe main.py data download
.\.venv\Scripts\python.exe main.py data catalog
.\.venv\Scripts\python.exe main.py data preprocess
.\.venv\Scripts\python.exe main.py data split

# One-fold, one-epoch smoke test
.\.venv\Scripts\python.exe main.py train --run-id pilot --models svm cnn rnn plain_snn st_snn --seeds 0 --max-folds 1 --epochs 1

# Five folds for one seed; no --max-folds means all configured folds
.\.venv\Scripts\python.exe main.py train --run-id final_cv5_seed0 --models svm cnn rnn plain_snn st_snn --seeds 0 --epochs 15

# Five folds for each of five seeds (25 folds per model)
.\.venv\Scripts\python.exe main.py train --run-id final_cv5_all_seeds --models svm cnn rnn plain_snn st_snn --seeds 0 1 2 3 4 --epochs 15

# Re-run analysis/reporting from saved outputs
.\.venv\Scripts\python.exe main.py analyze --results-dir results\final_cv5_seed0
.\.venv\Scripts\python.exe main.py report --results-dir results\final_cv5_seed0
```

Do not reuse a run ID to combine unrelated training settings. The result store is resumable and skips model/fold combinations that already have test prediction files. Use a new run ID when changing device, epoch budget, architecture/training settings, or dataset/split version.

## 7. Source and configuration file guide

### Command line and experiment orchestration

| File | What happens there | Why it matters |
|---|---|---|
| `main.py` | Defines `data`, `train`, `analyze`, and `report` command groups and loads the corresponding configuration. | It is the human-facing entry point and connects command options to the pipeline. |
| `experiments/run_experiment.py` | Loads processed data and split JSON; verifies group separation; builds loaders; fits each model; predicts validation then test; stores metrics, plots, efficiency, status, and the final report. | It is the end-to-end experiment controller and determines what the experiment actually did. |
| `experiments/__init__.py` | Marks `experiments` as an importable Python package. | Allows `main.py` and tools to import the runner. |
| `src/spikeattn/__init__.py` | Package marker and package-level metadata. | Makes the installed project importable. |

### Data, labels, preprocessing, and splits

| File | Responsibility and how it helps |
|---|---|
| `src/spikeattn/data/mitdb_downloader.py` | Downloads MIT-BIH record files and annotations. Keeps acquisition separate from modeling. |
| `src/spikeattn/data/record_catalog.py` | Defines class order and patient-group identity; catalogs records, beat counts, paced records, and shared-subject records. Establishes the subject key used by splitting. |
| `src/spikeattn/data/label_mapping.py` | Maps WFDB annotation symbols into the five configured AAMI classes and applies the unmapped-symbol policy. |
| `src/spikeattn/data/patient_splitter.py` | Defines patient partitions, asserts no group overlap, creates nested train/validation/test partitions, and writes split JSONs. Prevents patient leakage. |
| `src/spikeattn/preprocessing/filters.py` | Implements the Butterworth ECG band-pass filter. Removes frequencies outside the configured band. |
| `src/spikeattn/preprocessing/segmentation.py` | Cuts fixed windows around annotation sample locations and tracks offsets. Converts continuous recordings into beat examples. |
| `src/spikeattn/preprocessing/pipeline.py` | Reads WFDB signal/annotation pairs, filters, maps symbols, segments beats, and writes compressed arrays, metadata, and processed facts. Produces the model-ready cache. |
| `src/spikeattn/preprocessing/normalization.py` | Implements a lead-wise normalizer whose fit method requires explicit training patient groups. It exists as a leakage-aware utility but is not called in the current experiment runner. |
| `src/spikeattn/encoding/base_encoder.py` | Defines the common interface for spike encoders. Keeps encoding interchangeable. |
| `src/spikeattn/encoding/delta_encoder.py` | Converts voltage changes into ON/OFF events for each lead, producing four temporal channels. Supplies event-like inputs to the SNNs. |
| `src/spikeattn/evaluation/cross_validation.py` | Evaluation-oriented cross-validation helpers/contracts. The main executable split creation currently uses `data/patient_splitter.py`. |
| `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/metadata/.gitkeep` | Keep the expected data directories in version control while downloaded and generated data remain local/ignored. |

### Models

| File | Responsibility and how it helps |
|---|---|
| `src/spikeattn/models/base_model.py` | Defines the common model API (name, probability prediction, parameter accounting) expected by evaluation. |
| `src/spikeattn/models/baselines/svm_baseline.py` | Extracts handcrafted ECG features, scales them on training data, fits a modified-Huber SGD margin estimator, calibrates probabilities, and returns five-class probabilities. Despite its name, the present estimator is linear SGD, not an RBF SVM. |
| `src/spikeattn/models/baselines/cnn_baseline.py` | Implements the 1D convolution baseline and probability prediction. Tests whether local waveform filters are sufficient. |
| `src/spikeattn/models/baselines/rnn_baseline.py` | Implements the bidirectional GRU sequence baseline. Tests a recurrent temporal representation. |
| `src/spikeattn/models/snn/lif.py` | Implements LIF membrane update, thresholding, reset, and optional learnable decay. Provides spiking neuron dynamics. |
| `src/spikeattn/models/snn/surrogate.py` | Supplies a hard-forward spike with a smooth surrogate backward derivative. Makes gradient training possible through a nondifferentiable threshold. |
| `src/spikeattn/models/snn/spiking_conv.py` | Combines temporal Conv1D, batch normalization, and LIF activation. Builds spiking feature blocks. |
| `src/spikeattn/models/snn/attention.py` | Implements temporal self-attention and inter-lead attention. Lets the ST-SNN select relevant time context and cross-lead information. |
| `src/spikeattn/models/snn/plain_snn.py` | SNN baseline with delta encoding and spiking convolution but without attention. Isolates the contribution of adding attention. |
| `src/spikeattn/models/snn/st_snn.py` | Assembles encoder, two spiking conv blocks, inter-lead attention, temporal attention, pooling, and five-class head. Implements the proposed architecture. |
| `src/spikeattn/models/__init__.py`, `models/baselines/__init__.py`, `models/snn/__init__.py` | Package and export markers for model namespaces. |

### Training and prediction

| File | Responsibility and how it helps |
|---|---|
| `src/spikeattn/training/losses.py` | Creates class-weighted cross-entropy from training class counts. Reduces the dominance of common classes during optimization. |
| `src/spikeattn/training/trainer.py` | Moves neural models/batches to the configured device, trains with AdamW and gradient clipping, evaluates on validation data, applies early stopping, and restores the best validation checkpoint. Keeps test data outside checkpoint selection. |
| `src/spikeattn/training/hpo.py` | Provides an optional hyperparameter optimization wrapper. It is not invoked in the reported pilot (`hpo_trials_per_model` is null). |
| `src/spikeattn/prediction/predictor.py` | Runs batched probability inference and constructs `PredictionBundle` with validated row alignment and identifiers. Creates the common output contract for all models. |
| `src/spikeattn/prediction/__init__.py`, `src/spikeattn/training/__init__.py` | Package markers for prediction/training modules. |

### Evaluation, saved results, plots, and report

| File | Responsibility and how it helps |
|---|---|
| `src/spikeattn/evaluation/model_evaluator.py` | Computes beat-level accuracy, balanced accuracy, macro/weighted scores, per-class metrics, confusion matrices, OVR ROC curves, macro ROC-AUC/PR-AUC, and patient-level summaries. Keeps evaluation schema consistent. |
| `src/spikeattn/evaluation/results_store.py` | Writes split-specific prediction files, metric tables, probability curve JSON, efficiency measurements, and manifests; supports fold resume checks. Makes analysis reproducible from artifacts. |
| `src/spikeattn/evaluation/efficiency_profiler.py` | Measures repeated inference latency, process RSS, peak RSS, and CUDA memory; computes parameter/weight storage; estimates energy and labels provenance. Separates observations from estimates. |
| `src/spikeattn/visualization/performance_plots.py` | Validates metric schemas and renders confusion matrices, balanced-accuracy/F1 comparisons, class comparisons, ROC and PR curves. Fixed class order is N/S/V/F/Q. |
| `src/spikeattn/visualization/inference_plots.py` | Renders confidence-interval/forest plots and paired-difference plots. Helps communicate uncertainty and paired effects. |
| `src/spikeattn/visualization/efficiency_plots.py` | Plots performance against latency or estimated energy. Supports trade-off discussion, not proof of device-level energy superiority. |
| `src/spikeattn/visualization/distribution_plots.py` | Plots class and patient distributions. Helps show imbalance and sample coverage before modeling. |
| `src/spikeattn/reporting/report_builder.py` | Adds run/time context and renders the analysis bundle to Markdown using Jinja. |
| `src/spikeattn/reporting/templates/report.md.j2` | Report layout for descriptions, tests, CIs, split-specific metrics, efficiency, and limitations. |
| `src/spikeattn/evaluation/__init__.py`, `src/spikeattn/visualization/__init__.py`, `src/spikeattn/reporting/__init__.py` | Package markers. |
| `src/spikeattn/data/__init__.py`, `preprocessing/__init__.py`, `encoding/__init__.py`, `statistics/__init__.py` | Package markers and, for statistics, public exports. They make the module folders importable; they do not run pipeline work by themselves. |

### ISPA/statistical modules

| File | What happens there and how it helps |
|---|---|
| `src/spikeattn/statistics/base.py` | Defines `Hypothesis`, `TestResult`, and `BaseStatisticalTest`; standardizes alpha, tails, units, validation, and serialized test outputs. |
| `src/spikeattn/statistics/experiment_analyzer.py` | Loads stored CSV artifacts and orchestrates descriptive summaries, paired assumptions/tests, Friedman, bootstrap intervals, effect sizes, corrections, efficiency relationships, and the report bundle. This is the ISPA control center. |
| `src/spikeattn/statistics/ttest.py` | Implements one-sample, independent, Welch, paired, and corrected-resampled t procedures. The paired test is the relevant parametric comparison of same-patient scores. |
| `src/spikeattn/statistics/test_selector.py` | Applies the configured Shapiro/outlier rule to paired differences and selects paired t or Wilcoxon. |
| `src/spikeattn/statistics/assumption_tests.py` | Supplies Shapiro-Wilk, Levene, outlier detection, and other diagnostics used by test selection and exploratory analysis. |
| `src/spikeattn/statistics/nonparametric_tests.py` | Implements signed-rank Wilcoxon, matched Friedman with pairwise post-hoc tests, and Kruskal-Wallis. Covers rank-based alternatives and multi-model matched testing. |
| `src/spikeattn/statistics/multiple_comparison.py` | Implements Holm, Bonferroni, and Benjamini-Hochberg p-value correction. The planned primary four comparisons use Holm. |
| `src/spikeattn/statistics/confidence_intervals.py` | Implements Wilson, normal-approximation, percentile/BCa bootstrap, and paired difference intervals. In the primary result, bootstrap units are patient-level scores. |
| `src/spikeattn/statistics/effect_size.py` | Computes Cohen’s d, paired `d_z`, Hedges’ g, rank-biserial correlation, Kendall’s W, and other effect summaries. Adds practical magnitude context. |
| `src/spikeattn/statistics/descriptive_statistics.py` | Summarizes central tendency and spread without making inferential claims. |
| `src/spikeattn/statistics/classification_metrics.py` | Computes confusion matrix components, per-class and averaged scores, and a classification report. |
| `src/spikeattn/statistics/roc.py` | General ROC/OVR helper API. The experiment evaluator currently calculates probability curves directly with scikit-learn. |
| `src/spikeattn/statistics/hypothesis_testing.py` | Higher-level hypothesis-test orchestration utilities. The experiment analyzer directly coordinates its primary tests. |
| `src/spikeattn/statistics/anova.py` | One-way and repeated-measures ANOVA implementations for suitable designs; not used in the pilot’s primary comparisons. |
| `src/spikeattn/statistics/chi_square.py` | Chi-square tests for categorical/contingency analyses; not the primary model-score test. |
| `src/spikeattn/statistics/mcnemar_test.py` | McNemar test for paired classification disagreement under appropriate paired prediction conditions; not used in the report’s primary balanced-accuracy comparisons. |
| `src/spikeattn/statistics/correlation.py` | Pearson/Spearman correlation helpers for exploratory associations. |
| `src/spikeattn/statistics/regression.py` | OLS regression helpers and diagnostics, including exploratory efficiency/performance associations. |
| `src/spikeattn/statistics/efficiency_analysis.py` | Validates, summarizes, estimates CMOS energy from operation counts, and joins efficiency with performance. Energy results are estimates. |
| `src/spikeattn/statistics/__init__.py` | Re-exports statistical classes/functions as a package API. |

### Utilities, configuration, dependencies, and documentation

| File | Responsibility |
|---|---|
| `src/spikeattn/utils/config.py` | Loads YAML and validates configuration structure and required values. |
| `src/spikeattn/utils/hashing.py` | Creates canonical configuration hashes and file hashes for cache/split traceability. |
| `src/spikeattn/utils/reproducibility.py` | Seeds Python, NumPy, and PyTorch sources of randomness where called. |
| `src/spikeattn/utils/manifest.py` | Writes run/dataset provenance manifests. |
| `src/spikeattn/utils/logging_setup.py` | Configures log formatting, levels, and optional log files. |
| `src/spikeattn/utils/__init__.py`, `exceptions.py` | Utility package marker and typed errors for invalid config, bad data, insufficient sample size, and leakage. |
| `configs/dataset.yaml` | Input records, annotation source, sampling/window/filter settings, paced exclusion, and split plan. |
| `configs/label_mapping.yaml` | Fixed class order, annotation-to-class mapping, and unmapped symbol policy. |
| `configs/experiment.yaml` | Fold count, seeds, CUDA device, and validation fraction. Training CLI flags can override epochs, seeds, and fold count. |
| `configs/analysis.yaml` | Alpha, practical-interest threshold, correction method, and paired-test selection rule. |
| `configs/efficiency.yaml` | Efficiency-related settings/documentation for profiling. |
| `pyproject.toml` | Python 3.11 package metadata and exact runtime dependencies; configures the CUDA 12.4 PyTorch wheel source and test extra. |
| `requirements.txt` | Pip dependency list including the PyTorch CUDA wheel index. |
| `uv.lock` | Resolved dependency versions for repeatable uv installs. |
| `.gitignore` | Excludes local environment, generated data, and experiment results from source control. |
| `.env.example` | Placeholder for optional environment variables; it contains example names rather than secrets. |
| `README.md` | Project overview and quick documentation; some opening text still says design phase and should be reconciled with the implemented state. |
| `prd.md` | Product/research requirements and planned acceptance criteria; consult alongside the implementation, noting where the latter has evolved. |
| `arch.md`, `arch.mmd` | Architecture prose and diagram. Useful for conceptual structure; some details may describe earlier plans. |
| `parts.md` | Historical implementation plan/decisions. It is design history, not necessarily an exact description of current runtime behavior. |
| `IMPLEMENTATION_KICKOFF_PROMPT.md` | Original implementation scope and constraints. |
| `report.md` | This reader-oriented project, file, ISPA, and outcome guide. |

### Test files

Tests are executable checks, not research results. They validate behavior on synthetic or miniature data and do not establish performance on MIT-BIH.

| Test path | Main concern checked |
|---|---|
| `tests/conftest.py` | Shared synthetic ECG fixtures. |
| `tests/unit/test_config.py` | Configuration loading and validation. |
| `tests/unit/test_policy_terminology.py` | Probability-metric naming and documented averaging. |
| `tests/unit/data/test_record_catalog.py` | Patient-group catalog and record relationships. |
| `tests/unit/data/test_patient_splitter.py` | Deterministic grouped split behavior and disjointness. |
| `tests/unit/data/test_label_mapping.py` | Annotation-symbol mapping. |
| `tests/unit/preprocessing/test_filters.py` | Filter behavior. |
| `tests/unit/preprocessing/test_segmentation.py` | Beat-window segmentation. |
| `tests/unit/preprocessing/test_normalization.py` | Training-group-only normalizer contract. |
| `tests/unit/encoding/test_delta_encoder.py` | Spike encoding shape and threshold behavior. |
| `tests/unit/models/test_baselines.py` | Baseline model contracts. |
| `tests/unit/models/test_lif.py` | LIF dynamics. |
| `tests/unit/models/test_surrogate.py` | Surrogate-gradient behavior. |
| `tests/unit/models/test_attention.py` | Temporal/inter-lead attention shapes and behavior. |
| `tests/unit/models/test_st_snn.py` | ST-SNN forward/model contract. |
| `tests/unit/training/test_trainer.py` | Trainer/early-stopping behavior. |
| `tests/unit/prediction/test_predictor.py` | Prediction bundle construction and metadata validation. |
| `tests/unit/evaluation/test_results_store.py` | Persisted result schemas and resume checks. |
| `tests/unit/evaluation/test_efficiency_profiler.py` | Profiling metric schema and provenance kinds. |
| `tests/unit/evaluation/test_performance_plots.py` | Plotting with one/multiple models, absent classes, empty/malformed inputs. |
| `tests/unit/evaluation/test_model_evaluator_class_metadata.py` | Model identity and fixed class order in per-class metrics. |
| `tests/unit/statistics/test_ttest.py` | t-test statistics and results. |
| `tests/unit/statistics/test_test_selector.py` | Assumption-based test selection. |
| `tests/unit/statistics/test_nonparametric.py` | Wilcoxon/Friedman/Kruskal behavior. |
| `tests/unit/statistics/test_multiple_comparison.py` | Holm/Bonferroni/BH corrections. |
| `tests/unit/statistics/test_confidence_intervals.py` | Wilson, normal, and bootstrap intervals. |
| `tests/unit/statistics/test_effect_size.py` | Standardized and rank-based effects. |
| `tests/unit/statistics/test_assumption_tests.py` | Normality, variance, and outlier checks. |
| `tests/unit/statistics/test_metrics_and_roc.py` | Metric calculations and ROC helpers. |
| `tests/unit/statistics/test_descriptive.py` | Descriptive summaries. |
| `tests/unit/statistics/test_anova.py` | ANOVA procedures. |
| `tests/unit/statistics/test_chi_square.py` | Chi-square procedures. |
| `tests/unit/statistics/test_mcnemar.py` | Paired disagreement testing. |
| `tests/unit/statistics/test_hypothesis_testing.py` | Hypothesis-test orchestration. |
| `tests/unit/statistics/test_correlation.py` | Correlation calculations. |
| `tests/unit/statistics/test_regression.py` | Regression calculations. |
| `tests/unit/statistics/test_efficiency_analysis.py` | Efficiency summaries and estimates. |
| `tests/unit/statistics/__init__.py` | Marks the statistics test folder as a package. |
| `tests/integration/test_data_pipeline.py` | Data-flow integration on test fixtures. |
| `tests/integration/test_end_to_end_pipeline.py` | Miniature end-to-end run and result artifact integration. |
| `tests/integration/test_analyzer_synthetic_fixture.py` | ISPA analyzer behavior on synthetic results. |
| `tests/unit/reporting/test_report_builder.py` | Markdown rendering. |

## 8. How to read a run directory

Each run is written to `results/<run_id>/`:

| Artifact | Meaning |
|---|---|
| `splits/seed_<s>_fold_<f>.json` | Train, validation, and test patient-group assignments used for that fold. |
| `fold_status.csv` | Queued/running/completed/failed status per requested model/fold. Full-benchmark review should require all requested rows to be completed. |
| `predictions/validation/*.csv` | Validation beat probabilities/predictions, kept separate for model selection and diagnostics. |
| `predictions/test/*.csv` | Held-out fold test probabilities/predictions. These feed the reported test metrics. |
| `metrics/validation_metrics.csv`, `validation_per_class.csv`, `validation_confusion_matrices.csv` | Validation metrics, class detail, and confusion matrices. |
| `metrics/test_metrics.csv`, `test_per_class.csv`, `test_confusion_matrices.csv` | Corresponding held-out test artifacts. |
| `metrics/per_patient.csv` | Per-patient test balanced accuracy/accuracy/F1; primary input to patient-level comparisons. |
| `metrics/per_fold.csv`, `per_seed.csv` | Fold and seed summaries. A seed summary pools that seed’s out-of-fold predictions. |
| `metrics/pooled_oof.csv` | Combined out-of-fold predictions across folds; useful descriptive summary, **not an independent external test set**. |
| `probability_metrics/<split>/*.json` | OVR curve points, per-class probability metric status, and macro values. |
| `efficiency/measurements.csv` | Long-form measured/computed/estimated hardware and model metrics. Inspect `kind`, `method`, and `device`. |
| `visualizations/` | Confusion matrices, classification comparisons, ROC and precision-recall curves. |
| `report.md` | Rendered summary of analysis, intervals, metrics, and limitations. |

For the CUDA pilot, validation predictions contain 16,631 beats per model and test predictions contain 24,698 beats per model; the fold has nine test patient groups. This is the expected scope of fold 0, not all folds.

## 9. Honest conclusion and remaining work

The completed CUDA pilot demonstrates that the pipeline runs end-to-end, saves the promised split-separated artifacts, and executes the neural networks on the RTX 3050. Its predictive result does **not** establish that ST-SNN beats its baselines. In this pilot, CNN and RNN have higher held-out balanced accuracy than ST-SNN; the raw ST-SNN/RNN paired p-value does not survive Holm correction, and the matched Friedman result is nonsignificant. There are only nine patient observations, only one fold and one epoch, so uncertainty remains substantial.

Before making a research claim, run all five folds (and declare whether one or five seeds are being used), ensure every model/fold status is complete, verify split assignment and class support, review per-class results and confusion matrices, and report the patient-level matched analysis. Do not select the final model based on the test metrics. The report should describe the SVM accurately as implemented, correct the raw-versus-adjusted interpretation wording, identify inference timing as a 10-beat batch, and clearly separate CPU SVM timing from CUDA neural-model timing.

The most defensible current message is: **the project’s reproducible patient-grouped evaluation pipeline is operational; the one-fold pilot is inconclusive and provides no multiplicity-adjusted evidence of ST-SNN superiority.**
