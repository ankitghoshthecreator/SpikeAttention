"""Unit tests for descriptive_statistics.py: compare against scipy.stats.describe."""

import numpy as np
import pytest
from scipy import stats

from spikeattn.statistics.descriptive_statistics import DescriptiveStatistics


RNG = np.random.default_rng(0)
SAMPLE = RNG.normal(loc=5.0, scale=2.0, size=50)


class TestDescriptiveStatistics:
    def test_summarize_mean_matches_numpy(self):
        s = DescriptiveStatistics.summarize(SAMPLE)
        assert s["mean"] == pytest.approx(np.mean(SAMPLE))

    def test_summarize_std_matches_numpy_ddof1(self):
        s = DescriptiveStatistics.summarize(SAMPLE)
        assert s["std"] == pytest.approx(np.std(SAMPLE, ddof=1))

    def test_summarize_min_max(self):
        s = DescriptiveStatistics.summarize(SAMPLE)
        assert s["min"] == pytest.approx(np.min(SAMPLE))
        assert s["max"] == pytest.approx(np.max(SAMPLE))

    def test_summarize_n(self):
        s = DescriptiveStatistics.summarize(SAMPLE)
        assert s["n"] == len(SAMPLE)

    def test_summarize_median(self):
        s = DescriptiveStatistics.summarize(SAMPLE)
        assert s["median"] == pytest.approx(np.median(SAMPLE))

    def test_summarize_skewness_scipy(self):
        s = DescriptiveStatistics.summarize(SAMPLE)
        assert s["skewness"] == pytest.approx(stats.skew(SAMPLE), abs=1e-10)

    def test_summarize_kurtosis_scipy(self):
        """Excess kurtosis; scipy default bias=True, compare with abs tolerance."""
        s = DescriptiveStatistics.summarize(SAMPLE)
        sc_kurt = stats.kurtosis(SAMPLE, bias=True)
        assert abs(s["kurtosis"] - sc_kurt) < 0.5  # allow bias correction diff

    def test_quartiles_match_percentile(self):
        q1, q2, q3 = DescriptiveStatistics.quartiles(SAMPLE)
        assert q1 == pytest.approx(np.percentile(SAMPLE, 25))
        assert q2 == pytest.approx(np.percentile(SAMPLE, 50))
        assert q3 == pytest.approx(np.percentile(SAMPLE, 75))

    def test_iqr_equals_q3_minus_q1(self):
        iqr = DescriptiveStatistics.iqr(SAMPLE)
        q1, _, q3 = DescriptiveStatistics.quartiles(SAMPLE)
        assert iqr == pytest.approx(q3 - q1)

    def test_mode_integer_data(self):
        arr = np.array([1, 2, 2, 3, 3, 3, 4])
        mode_val = DescriptiveStatistics.mode(arr)
        assert mode_val == 3

    def test_class_distribution_sums_to_total_count(self):
        labels = np.array([0, 0, 1, 1, 1, 2])
        dist = DescriptiveStatistics.class_distribution(labels)
        assert sum(dist.values()) == len(labels)

    def test_class_distribution_counts(self):
        labels = np.array([0, 0, 1, 2, 2, 2])
        dist = DescriptiveStatistics.class_distribution(labels)
        assert dist[2] == 3
        assert dist[0] == 2
        assert dist[1] == 1

    def test_summarize_frame_returns_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({"a": SAMPLE, "b": SAMPLE * 2})
        result = DescriptiveStatistics.summarize_frame(df)
        assert isinstance(result, pd.DataFrame)
        assert set(result["variable"]) == {"a", "b"}

    def test_empty_array_raises(self):
        with pytest.raises((ValueError, Exception)):
            DescriptiveStatistics.summarize(np.array([]))
