"""Unit tests for EfficiencyProfiler."""

import numpy as np
import pytest
import torch

from spikeattn.evaluation.efficiency_profiler import EfficiencyProfiler
from spikeattn.models.snn.st_snn import SpikeAttentionSNN


class TestEfficiencyProfiler:
    def test_profile_model_contract(self):
        model = SpikeAttentionSNN()
        sample_in = np.random.randn(2, 2, 360).astype(np.float32)

        df = EfficiencyProfiler.profile_model(
            model=model,
            sample_input=sample_in,
            seed=0,
            fold=0,
            n_warmup=1,
            n_runs=2,
            device="cpu",
        )

        required_cols = {"model", "seed", "fold", "metric", "value", "unit", "kind", "method", "device"}
        assert required_cols.issubset(set(df.columns))

        # Check valid kinds
        valid_kinds = {"measured", "computed", "estimated"}
        for k in df["kind"]:
            assert k in valid_kinds

        # Energy must strictly be 'estimated'
        energy_rows = df[df["metric"] == "energy_uJ"]
        assert len(energy_rows) == 1
        assert energy_rows["kind"].iloc[0] == "estimated"

        # Latency must be 'measured'
        latency_rows = df[df["metric"] == "latency_ms"]
        assert len(latency_rows) == 1
        assert latency_rows["kind"].iloc[0] == "measured"

        # Parameters must be 'computed'
        param_rows = df[df["metric"] == "params"]
        assert len(param_rows) == 1
        assert param_rows["kind"].iloc[0] == "computed"
        assert param_rows["value"].iloc[0] > 0

    def test_weight_memory_mb_present(self):
        """weight_memory_mb must be emitted as a 'computed' metric."""
        model = SpikeAttentionSNN()
        sample_in = np.random.randn(2, 2, 360).astype(np.float32)

        df = EfficiencyProfiler.profile_model(
            model=model,
            sample_input=sample_in,
            seed=0,
            fold=0,
            n_warmup=1,
            n_runs=2,
            device="cpu",
        )

        wm_rows = df[df["metric"] == "weight_memory_mb"]
        assert len(wm_rows) == 1, "weight_memory_mb metric must be present"
        assert wm_rows["kind"].iloc[0] == "computed"
        assert wm_rows["unit"].iloc[0] == "mb"
        # Neural model with >0 params must have >0 weight memory
        assert wm_rows["value"].iloc[0] > 0.0
