"""Master orchestrator for the ISPA statistical analysis pipeline.

Loads Results-v1 CSV artefacts, runs the full multi-step analysis (descriptive,
assumption checks, model comparisons, confidence intervals, effect sizes, multiple-
comparison corrections, regression, and efficiency accounting) and returns an
``AnalysisBundle`` dictionary.

ISOLATION RULE: This module must NOT import from spikeattn.data,
spikeattn.preprocessing, spikeattn.models, spikeattn.training,
spikeattn.prediction, or spikeattn.evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spikeattn.statistics.assumption_tests import AssumptionTester
from spikeattn.statistics.base import Hypothesis, TestResult
from spikeattn.statistics.confidence_intervals import ConfidenceIntervalEstimator
from spikeattn.statistics.correlation import CorrelationAnalyzer
from spikeattn.statistics.descriptive_statistics import DescriptiveStatistics
from spikeattn.statistics.effect_size import EffectSizeAnalyzer
from spikeattn.statistics.efficiency_analysis import EfficiencyAnalyzer
from spikeattn.statistics.multiple_comparison import MultipleComparisonCorrector
from spikeattn.statistics.nonparametric_tests import FriedmanTest, WilcoxonSignedRankTest
from spikeattn.statistics.regression import RegressionAnalyzer
from spikeattn.statistics.test_selector import TestSelector
from spikeattn.statistics.ttest import PairedTTest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants (Results-v1 contract, frozen)
# ---------------------------------------------------------------------------
_COL_MODEL = "model"
_COL_SEED = "seed"
_COL_FOLD = "fold"
_COL_PATIENT_GROUP = "patient_group_id"
_COL_N_BEATS = "n_beats"
_COL_BALANCED_ACCURACY = "balanced_accuracy"
_COL_ACCURACY = "accuracy"
_COL_MACRO_F1 = "macro_f1"

_REQUIRED_PATIENT_COLS = {
    _COL_MODEL,
    _COL_SEED,
    _COL_FOLD,
    _COL_PATIENT_GROUP,
    _COL_N_BEATS,
    _COL_BALANCED_ACCURACY,
    _COL_ACCURACY,
    _COL_MACRO_F1,
}

# Primary hypothesis set: ST-SNN versus each baseline.
# baseline models are compared in this order.
_SNN_MODEL = "st_snn"
_BASELINE_MODELS = ("svm", "cnn", "rnn", "plain_snn")


class ExperimentAnalyzer:
    """Orchestrates the complete ISPA statistical analysis on a Results-v1 artefact tree.

    Loads per-patient, per-fold, per-seed, pooled, and efficiency CSV files,
    then executes a pre-registered analysis plan:

    1. Descriptive statistics for every model's balanced accuracy.
    2. Assumption checks (Shapiro-Wilk + IQR outlier rule) on paired differences.
    3. Primary model comparisons: ST-SNN vs each baseline with TestSelector.
    4. Patient-clustered bootstrap confidence intervals.
    5. Effect sizes (Hedges' g or rank-biserial).
    6. Multiple-comparison correction (Holm for primary, BH for secondary).
    7. Friedman test (secondary: omnibus across all models).
    8. Spearman correlation and OLS regression: cost vs performance.
    9. Efficiency accounting summary.

    Args:
        results_dir: Root directory of the Results-v1 artefact tree.
        config: Dictionary containing analysis configuration (e.g. alpha, ci_level,
                n_bootstrap, seed).
    """

    def __init__(self, results_dir: Path | str, config: dict[str, Any]) -> None:
        self._results_dir = Path(results_dir)
        self._config = config
        self._alpha: float = float(config.get("alpha", 0.05))
        self._ci_level: float = float(config.get("ci_level", 0.95))
        self._n_bootstrap: int = int(config.get("n_bootstrap", 2000))
        self._seed: int = int(config.get("seed", 42))
        self._primary_metric: str = config.get("primary_metric", _COL_BALANCED_ACCURACY)

        # Loaded frames — populated by load()
        self._per_patient: pd.DataFrame = pd.DataFrame()
        self._per_fold: pd.DataFrame = pd.DataFrame()
        self._per_seed: pd.DataFrame = pd.DataFrame()
        self._pooled: pd.DataFrame = pd.DataFrame()
        self._efficiency: pd.DataFrame = pd.DataFrame()

        self._loaded = False

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _csv(self, rel_path: str) -> Path:
        return self._results_dir / rel_path

    def load(self) -> None:
        """Read all Results-v1 CSV files from results_dir into DataFrames.

        Files are optional except per_patient.csv (required for primary inference).
        Missing optional files produce empty DataFrames with a logged warning.

        Raises:
            FileNotFoundError: If the mandatory per_patient.csv file is absent.
            ValueError: If per_patient.csv is missing required columns.
        """
        patient_path = self._csv("metrics/per_patient.csv")
        if not patient_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {patient_path}. "
                "Ensure Part 4 has written Results-v1 artefacts."
            )

        self._per_patient = pd.read_csv(patient_path)
        missing_cols = _REQUIRED_PATIENT_COLS - set(self._per_patient.columns)
        if missing_cols:
            raise ValueError(
                f"per_patient.csv is missing required columns: {missing_cols}"
            )

        for attr, path_str in [
            ("_per_fold", "metrics/per_fold.csv"),
            ("_per_seed", "metrics/per_seed.csv"),
            ("_pooled", "metrics/pooled.csv"),
        ]:
            p = self._csv(path_str)
            if p.exists():
                setattr(self, attr, pd.read_csv(p))
            else:
                logger.warning("Optional file missing (skipped): %s", p)

        eff_path = self._csv("efficiency/measurements.csv")
        if eff_path.exists():
            eff_df = pd.read_csv(eff_path)
            try:
                EfficiencyAnalyzer.validate_measurements_frame(eff_df)
                self._efficiency = eff_df
            except ValueError as exc:
                logger.warning("Efficiency measurements validation failed: %s", exc)
        else:
            logger.warning("Optional file missing (skipped): %s", eff_path)

        self._loaded = True
        logger.info(
            "Loaded Results-v1: %d patient-level rows, efficiency frame shape=%s",
            len(self._per_patient),
            self._efficiency.shape,
        )

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "Call ExperimentAnalyzer.load() before running analysis methods."
            )

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def describe(self) -> pd.DataFrame:
        """Compute descriptive statistics for the primary metric per model.

        Returns:
            DataFrame with one row per model summarising balanced accuracy.
        """
        self._require_loaded()
        metric = self._primary_metric
        rows = []
        for model, grp in self._per_patient.groupby(_COL_MODEL):
            vals = grp[metric].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue
            summary = DescriptiveStatistics.summarize(vals)
            summary["model"] = model
            rows.append(summary)
        return pd.DataFrame(rows)

    def check_assumptions(self) -> dict[str, dict[str, Any]]:
        """Check parametric test assumptions on paired differences for each comparison.

        For each baseline model, computes per-patient differences (ST-SNN minus baseline)
        and runs Shapiro-Wilk normality and IQR outlier checks.

        Returns:
            Mapping of ``baseline_name -> assumption check results dict``.
        """
        self._require_loaded()
        metric = self._primary_metric
        results: dict[str, dict[str, Any]] = {}

        snn_vals = self._patient_metric_vector(_SNN_MODEL, metric)
        if snn_vals is None:
            logger.warning("No data for primary model '%s'; skipping assumptions.", _SNN_MODEL)
            return results

        for baseline in _BASELINE_MODELS:
            base_vals = self._patient_metric_vector(baseline, metric)
            if base_vals is None:
                continue

            aligned_snn, aligned_base = self._aligned_pair(_SNN_MODEL, baseline, metric)
            diffs = aligned_snn - aligned_base
            if len(diffs) < 3:
                results[baseline] = {"skipped": True, "reason": f"n={len(diffs)} < 3"}
                continue

            norm = AssumptionTester.shapiro_wilk(diffs, alpha=self._alpha)
            outlier = AssumptionTester.outliers(diffs, method="iqr", threshold=3.0)
            results[baseline] = {
                "normality": norm,
                "outliers": outlier,
                "n_pairs": int(len(diffs)),
            }

        return results

    def compare_models(self) -> list[TestResult]:
        """Run primary family of model comparison tests (ST-SNN vs each baseline).

        Pre-specified decision rule: TestSelector chooses paired t-test or Wilcoxon.
        Applies Holm-Bonferroni multiple comparison correction.

        Returns:
            List of TestResult objects, one per comparison, with Holm-corrected p-values
            stored in a mutated TestResult's ``assumptions`` dict.
        """
        self._require_loaded()
        metric = self._primary_metric
        raw_results: list[TestResult] = []
        raw_p_values: list[float] = []

        snn_vals = self._patient_metric_vector(_SNN_MODEL, metric)
        if snn_vals is None:
            logger.warning("No data for '%s'; comparisons skipped.", _SNN_MODEL)
            return []

        hyp = Hypothesis(
            h0_text="No difference in balanced accuracy between ST-SNN and baseline.",
            h1_text="ST-SNN and baseline differ in balanced accuracy.",
            tail="two-sided",
            alpha=self._alpha,
        )

        for baseline in _BASELINE_MODELS:
            base_vals = self._patient_metric_vector(baseline, metric)
            if base_vals is None:
                logger.info("Baseline '%s' absent; skipping comparison.", baseline)
                continue

            aligned_snn, aligned_base = self._aligned_pair(_SNN_MODEL, baseline, metric)
            diffs = aligned_snn - aligned_base
            if len(diffs) < 3:
                logger.info("Too few pairs (%d) for '%s'; skipping.", len(diffs), baseline)
                continue

            selector = TestSelector.select_paired(diffs, alpha_assumption=self._alpha)
            if selector["selected_test"] == "paired_t":
                test = PairedTTest()
                result = test.run(aligned_snn, aligned_base, hypothesis=hyp, unit="patient_group")
            else:
                test = WilcoxonSignedRankTest()
                result = test.run(aligned_snn, aligned_base, hypothesis=hyp, unit="patient_group")

            raw_results.append(result)
            raw_p_values.append(result.p_value)

        if not raw_results:
            return []

        # Apply Holm correction across the primary family
        adj_p = MultipleComparisonCorrector.holm(raw_p_values)
        corrected: list[TestResult] = []
        for res, p_adj in zip(raw_results, adj_p):
            updated_assumptions = dict(res.assumptions)
            updated_assumptions["holm_corrected_p"] = float(p_adj)
            updated_assumptions["holm_reject_h0"] = bool(p_adj < self._alpha)
            # Build updated TestResult with corrected assumption info (re-use interpret text)
            corrected.append(
                TestResult(
                    name=res.name,
                    statistic=res.statistic,
                    df=res.df,
                    p_value=res.p_value,
                    alpha=res.alpha,
                    reject_h0=res.reject_h0,
                    critical_region=res.critical_region,
                    effect_size=res.effect_size,
                    ci=res.ci,
                    n=res.n,
                    unit=res.unit,
                    assumptions=updated_assumptions,
                    interpretation=res.interpretation,
                    valid=res.valid,
                )
            )

        return corrected

    def friedman_secondary(self) -> TestResult | None:
        """Omnibus Friedman test across all models (secondary analysis).

        Returns:
            TestResult if at least 3 models have paired data, else None.
        """
        self._require_loaded()
        metric = self._primary_metric
        all_models = [_SNN_MODEL] + list(_BASELINE_MODELS)
        vectors = []
        for model in all_models:
            v = self._patient_metric_vector(model, metric)
            if v is not None:
                vectors.append((model, v))

        if len(vectors) < 3:
            logger.info("Fewer than 3 models available; Friedman test skipped.")
            return None

        # Find common patient_group_ids (aggregating across seeds/folds)
        common_ids = None
        vec_map = {}
        for model, _ in vectors:
            pg_series = (
                self._per_patient[self._per_patient[_COL_MODEL] == model]
                .groupby(_COL_PATIENT_GROUP)[metric]
                .mean()
            )
            vec_map[model] = pg_series
            if common_ids is None:
                common_ids = set(pg_series.index)
            else:
                common_ids &= set(pg_series.index)

        if not common_ids or len(common_ids) < 3:
            return None

        if len(common_ids) != min(len(v) for _, v in vectors):
            logger.info("Friedman test uses %d matched patients across %d available models.", len(common_ids), len(vectors))

        common_ids_sorted = sorted(common_ids)
        groups = [
            vec_map[m].reindex(common_ids_sorted).to_numpy(dtype=np.float64)
            for m, _ in vectors
            if m in vec_map
        ]

        hyp = Hypothesis(
            h0_text="All models perform identically on balanced accuracy.",
            h1_text="At least one model differs significantly from the others.",
            tail="two-sided",
            alpha=self._alpha,
        )
        friedman = FriedmanTest()
        return friedman.run(*groups, hypothesis=hyp, unit="patient_group")

    def intervals(self) -> dict[str, dict[str, Any]]:
        """Compute patient-clustered bootstrap confidence intervals per model.

        Returns:
            Mapping of model name -> CI dict with keys ``lower``, ``upper``, ``point``,
            ``level``, ``method``.
        """
        self._require_loaded()
        metric = self._primary_metric
        ci_results: dict[str, dict[str, Any]] = {}

        for model in [_SNN_MODEL] + list(_BASELINE_MODELS):
            vals = self._patient_metric_vector(model, metric)
            if vals is None or len(vals) < 2:
                continue

            lower, upper = ConfidenceIntervalEstimator.bootstrap(
                vals,
                stat_fn=np.mean,
                level=self._ci_level,
                B=self._n_bootstrap,
                seed=self._seed,
            )
            ci_results[model] = {
                "point": float(np.mean(vals)),
                "lower": lower,
                "upper": upper,
                "level": self._ci_level,
                "method": "bootstrap",
            "n": int(len(vals)),
                "n_bootstrap": self._n_bootstrap,
            }

        return ci_results

    def effects(self) -> dict[str, dict[str, Any]]:
        """Compute effect sizes for each ST-SNN vs baseline comparison.

        Uses Hedges' g if paired t-test was selected, rank-biserial otherwise.

        Returns:
            Mapping of ``baseline_name -> effect_size dict``.
        """
        self._require_loaded()
        metric = self._primary_metric
        effect_results: dict[str, dict[str, Any]] = {}

        snn_vals = self._patient_metric_vector(_SNN_MODEL, metric)
        if snn_vals is None:
            return effect_results

        for baseline in _BASELINE_MODELS:
            base_vals = self._patient_metric_vector(baseline, metric)
            if base_vals is None:
                continue

            aligned_snn, aligned_base = self._aligned_pair(_SNN_MODEL, baseline, metric)
            diffs = aligned_snn - aligned_base
            if len(diffs) < 3:
                continue

            selector = TestSelector.select_paired(diffs, alpha_assumption=self._alpha)
            if selector["selected_test"] == "paired_t":
                g = EffectSizeAnalyzer.hedges_g(diffs)
                label = EffectSizeAnalyzer.interpret(g, metric="g")
                effect_results[baseline] = {
                    "effect_size": g,
                    "metric": "hedges_g",
                    "label": label,
                    "n_pairs": int(len(diffs)),
                }
            else:
                r_rb = EffectSizeAnalyzer.rank_biserial(diffs)
                label = EffectSizeAnalyzer.interpret(r_rb, metric="rank_biserial")
                effect_results[baseline] = {
                    "effect_size": r_rb,
                    "metric": "rank_biserial",
                    "label": label,
                    "n_pairs": int(len(diffs)),
                }

        return effect_results

    def correct(self) -> dict[str, Any]:
        """Apply multiple-comparison corrections to both primary and secondary families.

        Returns:
            Dict with keys ``primary`` (Holm-corrected) and ``secondary`` (BH-corrected).
        """
        self._require_loaded()
        comparisons = self.compare_models()

        primary_p = [r.p_value for r in comparisons]
        primary_adj = MultipleComparisonCorrector.holm(primary_p).tolist()

        friedman_res = self.friedman_secondary()
        secondary: dict[str, Any] = {}
        if friedman_res is not None and hasattr(friedman_res, "assumptions"):
            # Post-hoc p-values extracted from Friedman assumptions if available
            posthoc_p = friedman_res.assumptions.get("post_hoc_p_values", [])
            if posthoc_p:
                secondary["posthoc_p_values_bh"] = MultipleComparisonCorrector.benjamini_hochberg(
                    posthoc_p
                ).tolist()
                secondary["posthoc_raw_p_values"] = posthoc_p

        return {
            "primary": {
                "method": "holm",
                "raw_p_values": primary_p,
                "adjusted_p_values": primary_adj,
                "reject_h0": [bool(p < self._alpha) for p in primary_adj],
            },
            "secondary": secondary,
        }

    def relationships(self) -> dict[str, Any]:
        """Analyse relationships between model efficiency and performance.

        Runs Spearman correlation and simple OLS regression on median latency vs
        balanced accuracy, using per-model aggregate summaries.

        Returns:
            Dict with keys ``correlation`` and ``regression``.
        """
        self._require_loaded()
        output: dict[str, Any] = {"correlation": {}, "regression": {}}

        if self._efficiency.empty:
            logger.info("No efficiency measurements; skipping relationship analysis.")
            return output

        if self._per_patient.empty:
            return output

        try:
            merged = EfficiencyAnalyzer.relate_to_performance(
                self._per_patient, self._efficiency
            )
        except Exception as exc:
            logger.warning("relate_to_performance failed: %s", exc)
            return output

        if merged.empty or len(merged) < 3:
            return output

        # Identify numeric columns other than balanced_accuracy for correlation
        numeric_cols = [c for c in merged.columns if c not in {_COL_MODEL, _COL_BALANCED_ACCURACY}]
        if not numeric_cols:
            return output

        metric_col = numeric_cols[0]  # first efficiency metric found
        x = merged[metric_col].to_numpy(dtype=np.float64)
        y = merged[_COL_BALANCED_ACCURACY].to_numpy(dtype=np.float64)

        if len(x) < 3:
            return output

        spearman_res = CorrelationAnalyzer.spearman(x, y)
        output["correlation"] = spearman_res

        try:
            reg_res = RegressionAnalyzer.fit_simple(x, y)
            output["regression"] = reg_res
        except Exception as exc:
            logger.warning("Regression fit failed: %s", exc)

        return output

    def efficiency(self) -> pd.DataFrame:
        """Summarise and compare efficiency measurements across models.

        Returns:
            Summary DataFrame from EfficiencyAnalyzer, empty if no measurements loaded.
        """
        self._require_loaded()
        if self._efficiency.empty:
            logger.info("No efficiency measurements; returning empty summary.")
            return pd.DataFrame()

        return EfficiencyAnalyzer.summarize(self._efficiency)

    def report(self) -> dict[str, Any]:
        """Execute the full analysis pipeline and return an AnalysisBundle dictionary.

        Keys:
            descriptive: Per-model descriptive statistics DataFrame (serialised to records).
            assumptions: Assumption check results per baseline.
            comparisons: Primary model comparison TestResults (as dicts).
            friedman: Secondary omnibus Friedman TestResult dict or None.
            intervals: Bootstrap confidence intervals per model.
            effects: Effect sizes per comparison.
            corrections: Multiple-comparison correction results.
            relationships: Correlation and regression summary.
            efficiency: Efficiency summary DataFrame (serialised to records).

        Returns:
            AnalysisBundle dict suitable for JSON serialisation or further processing.
        """
        self._require_loaded()

        descriptive_df = self.describe()
        assumptions = self.check_assumptions()
        comparisons = self.compare_models()
        friedman = self.friedman_secondary()
        cis = self.intervals()
        effs = self.effects()
        corrections = self.correct()
        relations = self.relationships()
        eff_summary = self.efficiency()

        posthoc = self.post_hoc_friedman(friedman)
        bundle: dict[str, Any] = {
            "descriptive": descriptive_df.to_dict(orient="records"),
            "assumptions": assumptions,
            "comparisons": [r.to_dict() for r in comparisons],
            "friedman": friedman.to_dict() if friedman is not None else None,
            "posthoc": posthoc,
            "intervals": cis,
            "paired_difference_intervals": self.paired_difference_intervals(),
            "effects": effs,
            "corrections": corrections,
            "relationships": relations,
            "efficiency": eff_summary.to_dict(orient="records"),
        }
        logger.info(
            "AnalysisBundle complete: %d comparisons, %d CI entries, %d effect entries.",
            len(comparisons),
            len(cis),
            len(effs),
        )
        return bundle

    def run(self) -> dict[str, Any]:
        """Convenience entry point: load then report.

        Returns:
            AnalysisBundle dict (see ``report()``).
        """
        self.load()
        return self.report()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _patient_metric_vector(
        self, model: str, metric: str
    ) -> np.ndarray | None:
        """Return per-patient metric values for a model, averaged over folds and seeds.

        Returns None if the model has no data.
        """
        sub = self._per_patient[self._per_patient[_COL_MODEL] == model]
        if sub.empty:
            return None
        agg = (
            sub.groupby(_COL_PATIENT_GROUP)[metric]
            .mean()
        )
        return agg.to_numpy(dtype=np.float64)

    def _align_paired(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align two patient metric vectors by their common patient_group_ids.

        Returns arrays of equal length ordered by sorted patient_group_id.
        """
        # Reconstruct Series indexed by patient_group_id for alignment
        def _series(model: str, metric: str) -> pd.Series:
            sub = self._per_patient[self._per_patient[_COL_MODEL] == model]
            return (
                sub.groupby(_COL_PATIENT_GROUP)[metric]
                .mean()
            )

        # Find out which model the two vectors came from by scanning
        # (re-compute for alignment purposes — this is O(n) and not in a hot loop)
        metric = self._primary_metric
        snn_s = _series(_SNN_MODEL, metric)

        # Identify baseline from vectors by matching length
        matched_model: str | None = None
        for bm in _BASELINE_MODELS:
            bs = _series(bm, metric)
            if len(bs) == len(b) and np.allclose(bs.to_numpy(dtype=np.float64), b, rtol=1e-5, atol=1e-8):
                matched_model = bm
                break

        if matched_model is None:
            # Fallback: truncate or pad to shortest common length
            n = min(len(a), len(b))
            return a[:n], b[:n]

        base_s = _series(matched_model, metric)
        common = sorted(set(snn_s.index) & set(base_s.index))
        if not common:
            n = min(len(a), len(b))
            return a[:n], b[:n]

        return (
            snn_s.reindex(common).to_numpy(dtype=np.float64),
            base_s.reindex(common).to_numpy(dtype=np.float64),
        )

    def _aligned_pair(self, model_a: str, model_b: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        """Align model outcomes explicitly by patient group ID."""
        def by_patient(model: str) -> pd.Series:
            return (self._per_patient[self._per_patient[_COL_MODEL] == model]
                    .groupby(_COL_PATIENT_GROUP)[metric].mean())
        a, b = by_patient(model_a), by_patient(model_b)
        common = sorted(set(a.index) & set(b.index))
        return a.reindex(common).to_numpy(dtype=np.float64), b.reindex(common).to_numpy(dtype=np.float64)

    def post_hoc_friedman(self, result: TestResult | None) -> list[dict[str, Any]]:
        """Run matched Wilcoxon pairwise post-hoc tests only after a significant omnibus test."""
        if result is None or result.p_value >= self._alpha:
            return []
        models = [_SNN_MODEL] + list(_BASELINE_MODELS)
        vectors = {m: self._per_patient[self._per_patient[_COL_MODEL] == m]
                   .groupby(_COL_PATIENT_GROUP)[self._primary_metric].mean() for m in models}
        common = sorted(set.intersection(*(set(v.index) for v in vectors.values())))
        matrix = np.column_stack([vectors[m].reindex(common).to_numpy(dtype=np.float64) for m in models])
        return FriedmanTest().post_hoc(matrix, models, method="holm").to_dict(orient="records")

    def paired_difference_intervals(self) -> dict[str, dict[str, Any]]:
        """Bootstrap mean matched patient-level balanced-accuracy differences."""
        output: dict[str, dict[str, Any]] = {}
        for baseline in _BASELINE_MODELS:
            a, b = self._aligned_pair(_SNN_MODEL, baseline, self._primary_metric)
            diffs = a - b
            if len(diffs) < 2:
                continue
            lower, upper = ConfidenceIntervalEstimator.bootstrap(
                diffs, stat_fn=np.mean, B=self._n_bootstrap, level=self._ci_level, seed=self._seed)
            output[baseline] = {"mean_difference": float(np.mean(diffs)), "lower": lower, "upper": upper,
                                "n_patients": int(len(diffs)), "method": "paired patient bootstrap percentile",
                                "n_bootstrap": self._n_bootstrap}
        return output
