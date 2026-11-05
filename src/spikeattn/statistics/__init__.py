"""ISPA Statistical Engine — Part 2 public API.

All public classes and dataclasses are re-exported here so that
downstream code (Part 4 only) can do::

    from spikeattn.statistics import ExperimentAnalyzer, TestResult, ...

ISOLATION RULE: This package must NOT import from spikeattn.data,
spikeattn.preprocessing, spikeattn.models, spikeattn.training,
spikeattn.prediction, or spikeattn.evaluation.
"""

from spikeattn.statistics.anova import OneWayANOVA, RepeatedMeasuresANOVA
from spikeattn.statistics.assumption_tests import AssumptionTester
from spikeattn.statistics.base import BaseStatisticalTest, Hypothesis, TestResult
from spikeattn.statistics.chi_square import ChiSquareTest
from spikeattn.statistics.classification_metrics import ClassificationMetrics
from spikeattn.statistics.confidence_intervals import ConfidenceIntervalEstimator
from spikeattn.statistics.correlation import CorrelationAnalyzer
from spikeattn.statistics.descriptive_statistics import DescriptiveStatistics
from spikeattn.statistics.effect_size import EffectSizeAnalyzer
from spikeattn.statistics.efficiency_analysis import EfficiencyAnalyzer
from spikeattn.statistics.experiment_analyzer import ExperimentAnalyzer
from spikeattn.statistics.hypothesis_testing import HypothesisTester
from spikeattn.statistics.mcnemar_test import McNemarTest
from spikeattn.statistics.multiple_comparison import MultipleComparisonCorrector
from spikeattn.statistics.nonparametric_tests import (
    FriedmanTest,
    KruskalWallisTest,
    WilcoxonSignedRankTest,
)
from spikeattn.statistics.regression import RegressionAnalyzer
from spikeattn.statistics.roc import ROCAnalyzer
from spikeattn.statistics.test_selector import TestSelector
from spikeattn.statistics.ttest import (
    CorrectedResampledTTest,
    IndependentTTest,
    OneSampleTTest,
    PairedTTest,
    WelchTTest,
)

__all__ = [
    # Base
    "Hypothesis",
    "TestResult",
    "BaseStatisticalTest",
    # Descriptive
    "DescriptiveStatistics",
    # Assumption checks
    "AssumptionTester",
    # T-tests
    "OneSampleTTest",
    "IndependentTTest",
    "WelchTTest",
    "PairedTTest",
    "CorrectedResampledTTest",
    # Non-parametric
    "WilcoxonSignedRankTest",
    "FriedmanTest",
    "KruskalWallisTest",
    # McNemar & Chi-square
    "McNemarTest",
    "ChiSquareTest",
    # ANOVA
    "OneWayANOVA",
    "RepeatedMeasuresANOVA",
    # Confidence intervals
    "ConfidenceIntervalEstimator",
    # Effect sizes
    "EffectSizeAnalyzer",
    # Multiple comparison
    "MultipleComparisonCorrector",
    # Hypothesis testing registry
    "HypothesisTester",
    # Correlation & regression
    "CorrelationAnalyzer",
    "RegressionAnalyzer",
    # Classification & ROC
    "ClassificationMetrics",
    "ROCAnalyzer",
    # Efficiency
    "EfficiencyAnalyzer",
    # Test selector
    "TestSelector",
    # Orchestrator
    "ExperimentAnalyzer",
]
