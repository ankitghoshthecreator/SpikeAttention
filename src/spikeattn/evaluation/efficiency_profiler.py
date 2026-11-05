"""EfficiencyProfiler: accounts hardware latency, memory, parameter counts, and estimated energy."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
import torch

from spikeattn.statistics.efficiency_analysis import EfficiencyAnalyzer


class EfficiencyProfiler:
    """Measures and computes computational complexity metrics strictly tagged by kind."""

    @staticmethod
    def profile_model(
        model: Any,
        sample_input: np.ndarray | torch.Tensor,
        seed: int = 0,
        fold: int = 0,
        n_warmup: int = 5,
        n_runs: int = 20,
        device: str = "cpu",
    ) -> pd.DataFrame:
        """Benchmark model execution and build measurements conforming to Results Contract v1.0.0.

        Headers:
            model, seed, fold, metric, value, unit, kind, method, device
        """
        model_name = getattr(model, "name", "unknown_model")
        rows: list[dict[str, Any]] = []

        # 1. Parameter Count (computed)
        param_count = float(getattr(model, "count_parameters", lambda: 0)())
        rows.append({
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "metric": "params",
            "value": param_count,
            "unit": "count",
            "kind": "computed",
            "method": "torch_parameter_count",
            "device": device,
        })

        # 2. Latency Measurement (measured)
        # Warmup
        for _ in range(n_warmup):
            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(sample_input)
            elif callable(model):
                _ = model(sample_input)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(sample_input)
            elif callable(model):
                _ = model(sample_input)
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        median_latency = float(np.median(latencies))
        rows.append({
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "metric": "latency_ms",
            "value": median_latency,
            "unit": "ms",
            "kind": "measured",
            "method": f"median_of_{n_runs}_runs_after_{n_warmup}_warmup",
            "device": device,
        })

        # 3. Peak RAM / Memory (measured)
        tracemalloc.start()
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(sample_input)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = float(peak_bytes / (1024 * 1024))
        rows.append({
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "metric": "peak_memory_mb",
            "value": peak_mb,
            "unit": "mb",
            "kind": "measured",
            "method": "tracemalloc_peak",
            "device": device,
        })

        # 4. Energy Estimate (ESTIMATED — NEVER REPORTED AS MEASURED)
        # Approximate MACs / SynOps based on model type
        if "snn" in model_name:
            # SNN operations are predominantly synaptic accumulates (AC)
            est_macs = int(param_count * 0.1)
            est_synops = int(param_count * 16)
        else:
            est_macs = int(param_count * 10)
            est_synops = 0

        energy_dict = EfficiencyAnalyzer.energy_estimate(macs=est_macs, synops=est_synops)
        rows.append({
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "metric": "energy_uJ",
            "value": float(energy_dict["total_energy_uJ"]),
            "unit": "uJ",
            "kind": "estimated",
            "method": "horowitz_cmos_energy_model",
            "device": device,
        })

        df = pd.DataFrame(rows)
        EfficiencyAnalyzer.validate_measurements_frame(df)
        return df
