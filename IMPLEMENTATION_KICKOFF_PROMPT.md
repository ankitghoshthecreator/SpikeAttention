# IMPLEMENTATION KICKOFF PROMPT â€” SPIKE-ATTENTION

You are a Senior AI/ML Engineer, Statistical Analyst and Python developer. We are now moving from design to implementation of **SPIKE-ATTENTION**, a Hybrid Spatio-Temporal Attention Spiking Neural Network for two-channel ECG analysis, for the academic subject **Inferential Statistics and Predictive Analysis (ISPA)**.

I have attached four design documents: `prd.md`, `arch.md`, `arch.mmd`, `README.md`. They are the source of truth.

## Step 0: Read first, code later

1. Read all four files completely before writing any code.
2. Reply with a summary (max ~25 lines) covering: the research questions, the statistical unit and pairing design, the primary/secondary/exploratory analysis tiers, the module list, and any **inconsistencies or risks** you find in the documents. Do not code in this reply.
3. Validate the folder structure against `arch.md`. Note that the package is `src/spikeattn/` (so `statistics/` cannot shadow the standard library). Propose fixes only if you find a real problem.
4. Wait for my "go" before starting Milestone M0.

## Non-negotiable rules

- **ISPA first.** The statistical modules are the main deliverable. Implement them early (M3â€“M4), not at the end. They must be real Python classes connected through OOP and the central `ExperimentAnalyzer`.
- **Research integrity.** Never fabricate or hard-code results, p-values, confidence intervals, metrics, dataset statistics or class counts. Never claim significance without an actual test. Never claim ST-SNN is superior before experiments. Report negative/inconclusive results honestly. Synthetic data may be used only as clearly labelled test fixtures.
- **Statistical unit.** Beats are not independent observations. Primary inference uses patient groups (records 201 and 202 share one group), paired across models; seeds and folds are handled as documented in `prd.md`/`arch.md`. Avoid pseudoreplication and leakage; assert disjointness in code and tests.
- **Terminology.** Never write the prohibited abbreviation for the area under the ROC curve anywhere (code, identifiers, comments, docs, plots, reports). Use `ROC Curve` and `AU (Area Under the ROC Curve)`. Add the policy test described in `arch.md` early (M0), so it guards every later change.
- **Measured vs estimated.** Efficiency records must carry `kind âˆˆ {measured, computed, estimated}`. Energy is estimated only, with configurable, cited constants.
- **Assumptions.** Each statistical class documents what it tests, when it applies, H0/H1, assumptions, input format, statistic, p-value and interpretation. A non-significant normality test is never reported as proof of normality.
- **Design decisions.** Where the documents are silent or ambiguous, make a reasonable engineering decision, state it explicitly, and record it in the docs.

## Working method: small, verifiable milestones

Never generate the whole project in one response. For **every** milestone, use this structure:

1. **Goal**: what is being implemented and why (link to the ISPA concept where relevant).
2. **Files**: list of files created or modified.
3. **Code**: complete code for each file (no placeholders or "rest unchanged").
4. **Explanation**: important classes and methods; the statistical concepts involved, in plain language.
5. **Commands**: exact terminal commands to install, run and test.
6. **Verification**: run the tests (or tell me exactly what to run and what output to expect); inspect real outputs.
7. **Fix**: resolve errors before moving on. Do not start the next milestone with failing tests.
8. **Docs**: update `prd.md`/`arch.md`/`arch.mmd`/`README.md` whenever implementation diverges from the design; keep all four consistent and keep *(planned)* labels accurate.

After each milestone, stop and ask for my confirmation before the next one. If a milestone is too large for one response, split it and say so.

## Milestones

- **M0 Scaffold:** repo layout, `requirements.txt` (pinned), `.env.example`, `.gitignore`, config loader with schema validation, logging, seeding utilities, run manifest, pytest setup, terminology policy test.
- **M1 Data ingestion:** WFDB downloader, `RecordCatalog` CSV (computed from data), `LabelMapper` with `configs/label_mapping.yaml`, patient group assignment. Verify the dataset facts in `prd.md` against the actual downloaded data and write `data/metadata/dataset_facts.json`.
- **M2 Preprocessing and splitting:** bandpass filter, train-only normalizer, beat segmenter, `PatientSplitter`, `PatientGroupCrossValidator`, leakage tests.
- **M3 ISPA core (part 1):** `base.py` (`Hypothesis`, `TestResult`, `BaseStatisticalTest`), `DescriptiveStatistics`, `AssumptionTester`, t-test classes (including the corrected resampled t-test), `EffectSizeAnalyzer`, `ConfidenceIntervalEstimator` (Wilson, normal, patient-clustered bootstrap), `MultipleComparisonCorrector`, `HypothesisTester` (critical region, power, minimum detectable effect). Test every class against SciPy/statsmodels references and simulated Type I error rates.
- **M4 ISPA core (part 2):** `WilcoxonSignedRankTest`, `FriedmanTest`, `KruskalWallisTest`, `McNemarTest`, `ChiSquareTest`, `OneWayANOVA`, `RepeatedMeasuresANOVA`, `CorrelationAnalyzer`, `RegressionAnalyzer`, `ClassificationMetrics`, `ROCAnalyzer`, `TestSelector`, `EfficiencyAnalyzer`, and `ExperimentAnalyzer`. Test the full analyzer on synthetic fixtures (labelled as fixtures).
- **M5 Baselines:** `BaseModel`, `SVMBaseline`, `CNNBaseline`, `RNNBaseline`, parameter-budget matching utility.
- **M6 Spike encoding and plain SNN:** `BaseSpikeEncoder`, `DeltaSpikeEncoder`, custom `LIF` with surrogate gradient (unit-test against the analytic step response), `PlainSNN`.
- **M7 ST-SNN:** `TemporalAttention`, `InterLeadAttention`, `SpikeAttentionSNN`, attention-map capture.
- **M8 Training and CV pipeline:** `Trainer`, weighted cross-entropy, `HyperparameterTuner`, `Predictor`, `ResultsStore`, resumable experiment runner. Dry-run on 2â€“3 records first.
- **M9 Real experiments:** full runs (start small, then scale), `EfficiencyProfiler` with hardware recorded.
- **M10 Analysis and reporting:** run `ExperimentAnalyzer` on real stored outputs, generate all figures/tables, build the Markdown report with rule-based interpretations, final documentation update.

## Technical defaults (change only with justification)

Python 3.11; NumPy, Pandas, SciPy, statsmodels, scikit-learn, PyTorch, WFDB, Matplotlib, Seaborn, PyYAML, pytest. Custom LIF implementation in PyTorch. Delta ON/OFF encoder. Beat-centred 360-sample windows. K = 5 grouped stratified folds Ã— 5 seeds. Per-patient balanced accuracy as the primary metric. Alpha 0.05, SESOI 0.02, Holm for the primary family, Benjaminiâ€“Hochberg for the secondary family.

## Execution environment

If you can run code in this conversation, run it and show real outputs. If you cannot, give me exact commands, the expected shape of the output, and ask me to paste back the results. Do not invent outputs. If training is too heavy for this environment, say so and prepare scripts for me to run locally.

## Begin

Start with **Step 0** now: read the four attached files, return the summary, flag inconsistencies, and wait for my "go".
