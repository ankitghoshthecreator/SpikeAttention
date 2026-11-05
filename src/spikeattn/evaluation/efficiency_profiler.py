"""Consistent process-level inference profiling with explicit metric provenance."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

from spikeattn.statistics.efficiency_analysis import EfficiencyAnalyzer


class EfficiencyProfiler:
    @staticmethod
    def profile_model(model: Any, sample_input: np.ndarray | torch.Tensor, seed: int = 0, fold: int = 0,
                      n_warmup: int = 5, n_runs: int = 30, device: str = "cpu") -> pd.DataFrame:
        if n_runs < 2 or n_warmup < 0:
            raise ValueError("Use at least two measured runs and a non-negative warm-up count")
        name = getattr(model, "name", "unknown_model")
        protocol = f"same_{len(sample_input)}_examples;{n_warmup}_warmup;{n_runs}_runs"
        rows: list[dict[str, Any]] = []
        def add(metric: str, value: float, unit: str, kind: str, method: str) -> None:
            rows.append({"model": name, "seed": seed, "fold": fold, "metric": metric, "value": value,
                         "unit": unit, "kind": kind, "method": method, "device": device})
        param_count = float(getattr(model, "count_parameters", lambda: 0)())
        sklearn_state_bytes = 0
        if hasattr(model, "clf") and hasattr(model.clf, "coef_"):
            estimator = model.clf
            arrays = [getattr(estimator, key) for key in ("coef_", "intercept_") if hasattr(estimator, key)]
            sklearn_state_bytes += sum(int(array.nbytes) for array in arrays)
            param_count = float(sum(int(array.size) for array in arrays))
            scaler = getattr(model, "scaler", None)
            if scaler is not None:
                sklearn_state_bytes += sum(int(getattr(scaler, key).nbytes) for key in ("mean_", "scale_") if hasattr(scaler, key))
            calibration = getattr(model, "calibrated", None)
            for calibrated in getattr(calibration, "calibrated_classifiers_", []):
                for calibrator in getattr(calibrated, "calibrators", []):
                    sklearn_state_bytes += sum(int(np.asarray(getattr(calibrator, key)).nbytes)
                        for key in ("a_", "b_") if hasattr(calibrator, key))
        add("params", param_count, "count", "computed", "model_parameter_count")
        def infer() -> None:
            if hasattr(model, "predict_proba"):
                model.predict_proba(sample_input)
            elif callable(model):
                with torch.inference_mode(): model(sample_input)
            else:
                raise TypeError(f"{name} has no inference interface")
        for _ in range(n_warmup): infer()
        proc = psutil.Process()
        rss_samples = [proc.memory_info().rss]
        stop = threading.Event()
        def sample_rss() -> None:
            while not stop.is_set():
                try: rss_samples.append(proc.memory_info().rss)
                except psutil.Error: pass
                stop.wait(0.005)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        thread = threading.Thread(target=sample_rss, daemon=True); thread.start()
        latencies = []
        try:
            for _ in range(n_runs):
                t0 = time.perf_counter(); infer()
                if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)
                rss_samples.append(proc.memory_info().rss)
        finally:
            stop.set(); thread.join()
        add("latency_mean_ms", float(np.mean(latencies)), "ms", "measured", protocol)
        add("latency_median_ms", float(np.median(latencies)), "ms", "measured", protocol)
        add("latency_ms", float(np.median(latencies)), "ms", "measured", protocol + ";legacy_median_alias")
        add("latency_p95_ms", float(np.percentile(latencies, 95)), "ms", "measured", protocol)
        add("process_rss_mb", float(rss_samples[0] / 2**20), "mb", "measured", "psutil_process_rss_before_inference")
        add("peak_process_rss_mb", float(max(rss_samples) / 2**20), "mb", "measured", "psutil_5ms_poll_during_inference")
        if device.startswith("cuda") and torch.cuda.is_available():
            add("gpu_peak_memory_mb", float(torch.cuda.max_memory_allocated() / 2**20), "mb", "measured", "torch_cuda_peak_allocated")
        if sklearn_state_bytes:
            weight_bytes = sklearn_state_bytes
            method = "sklearn_coefficients_intercepts_scaler_and_calibration_arrays_actual_nbytes"
        else:
            weight_bytes = param_count * 4.0; method = "parameter_count_x_float32_bytes"
        add("weight_memory_mb", weight_bytes / 2**20, "mb", "computed", method)
        if "snn" in name:
            macs, synops = int(param_count * .1), int(param_count * 16)
        else:
            macs, synops = int(param_count * 10), 0
        energy = EfficiencyAnalyzer.energy_estimate(macs=macs, synops=synops)
        add("energy_uJ", float(energy["total_energy_uJ"]), "uJ", "estimated", "horowitz_cmos_energy_model_not_physical_measurement")
        df = pd.DataFrame(rows); EfficiencyAnalyzer.validate_measurements_frame(df)
        return df
