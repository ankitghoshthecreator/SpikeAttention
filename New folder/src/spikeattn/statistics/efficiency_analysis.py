"""Computational efficiency accounting and analysis: separating measured, computed, and estimated quantities."""

from typing import Any

import numpy as np
import pandas as pd


class EfficiencyAnalyzer:
    """Analyzes hardware latency, parameters, memory, spike activity, and estimated energy."""

    VALID_KINDS = {"measured", "computed", "estimated"}

    @staticmethod
    def validate_measurements_frame(df: pd.DataFrame) -> None:
        """Validate that all records strictly declare kind in {measured, computed, estimated}."""
        required_cols = {"model", "metric", "value", "unit", "kind"}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Efficiency measurements DataFrame missing required columns: {missing}")

        invalid_kinds = set(df["kind"].unique()) - EfficiencyAnalyzer.VALID_KINDS
        if invalid_kinds:
            raise ValueError(f"Invalid efficiency measurement kind(s): {invalid_kinds}; must be in {EfficiencyAnalyzer.VALID_KINDS}")

    @staticmethod
    def summarize(df: pd.DataFrame) -> pd.DataFrame:
        """Summarize efficiency measurements by model and metric."""
        EfficiencyAnalyzer.validate_measurements_frame(df)

        rows = []
        for (model, metric), group in df.groupby(["model", "metric"]):
            vals = group["value"].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue

            q25, q75 = np.percentile(vals, [25, 75])
            kind = group["kind"].iloc[0]
            unit = group["unit"].iloc[0]

            rows.append({
                "model": model,
                "metric": metric,
                "unit": unit,
                "kind": kind,
                "n": int(len(vals)),
                "median": float(np.median(vals)),
                "iqr": float(q75 - q25),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def compare(df: pd.DataFrame, baseline: str = "plain_snn") -> pd.DataFrame:
        """Compute relative percentage difference against a designated baseline model."""
        summary = EfficiencyAnalyzer.summarize(df)
        if summary.empty:
            return pd.DataFrame()

        base_sub = summary[summary["model"] == baseline].set_index("metric")["median"].to_dict()

        diff_rows = []
        for _, row in summary.iterrows():
            metric = row["metric"]
            base_val = base_sub.get(metric)
            if base_val is not None and base_val > 0:
                rel_diff_pct = float(((row["median"] - base_val) / base_val) * 100.0)
            else:
                rel_diff_pct = None

            row_dict = row.to_dict()
            row_dict["baseline_median"] = base_val
            row_dict["relative_diff_pct"] = rel_diff_pct
            diff_rows.append(row_dict)

        return pd.DataFrame(diff_rows)

    @staticmethod
    def energy_estimate(
        macs: int,
        synops: int,
        e_mac: float = 4.6e-12,  # 4.6 pJ per 32-bit MAC (standard 45nm/28nm CMOS literature estimate)
        e_ac: float = 0.9e-12,   # 0.9 pJ per synaptic accumulate (AC)
        source: str = "Horowitz (2014) 28nm/45nm CMOS energy benchmark",
    ) -> dict[str, Any]:
        """Estimate per-inference energy from MAC and SynOp operations.

        CRITICAL INTEGRITY RULE: Energy is NEVER reported as measured; it carries kind='estimated'.
        """
        mac_energy = float(macs * e_mac)
        ac_energy = float(synops * e_ac)
        total_energy = mac_energy + ac_energy

        return {
            "kind": "estimated",
            "unit": "joules",
            "mac_energy_joules": mac_energy,
            "synop_energy_joules": ac_energy,
            "total_energy_joules": total_energy,
            "total_energy_uJ": total_energy * 1e6,
            "assumptions": {
                "e_mac_joules": e_mac,
                "e_ac_joules": e_ac,
                "literature_source": source,
                "note": "Excludes memory access overhead, I/O, and clock distribution energy",
            },
        }

    @staticmethod
    def relate_to_performance(metrics_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> pd.DataFrame:
        """Merge model accuracy metrics with latency/energy profiles for cost-benefit analysis."""
        perf_summary = metrics_df.groupby("model")["balanced_accuracy"].mean().reset_index()
        eff_summary = efficiency_df.groupby(["model", "metric"])["value"].median().unstack().reset_index()
        return pd.merge(perf_summary, eff_summary, on="model", how="inner")
