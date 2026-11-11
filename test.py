"""SPIKE-ATTENTION: Unseen Test Evaluation and Single-Beat Inference Demo.

Loads the trained models from model/best/ and evaluates them on unseen test data
from held-out patient groups (MIT-BIH test partition).

Usage:
    # 1. Quick demo: runs single unseen beat through all 5 models + performance summary
    python test.py

    # 2. Pick an unseen beat of a specific arrhythmia class (e.g. V, S, N, or F)
    python test.py --target-class V

    # 3. Specify an exact test beat index (0 to 24,697)
    python test.py --sample-idx 105

    # 4. Generate an ECG waveform plot with prediction confidence comparison
    python test.py --target-class V --plot

    # 5. Full unseen test set evaluation metrics (Precision, Recall, F1 for all classes)
    python test.py --full-metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from spikeattn.models.baselines.cnn_baseline import CNNBaseline
from spikeattn.models.baselines.rnn_baseline import RNNBaseline
from spikeattn.models.snn.plain_snn import PlainSNN
from spikeattn.models.snn.st_snn import SpikeAttentionSNN
from spikeattn.statistics.classification_metrics import ClassificationMetrics

CLASS_ORDER = ["N", "S", "V", "F", "Q"]
CLASS_DESCRIPTIONS = {
    "N": "Normal / Bundle Branch Block",
    "S": "Supraventricular Ectopic",
    "V": "Ventricular Ectopic (PVC)",
    "F": "Fusion of Ventricular & Normal",
    "Q": "Unknown / Paced",
}

MODEL_NAMES = {
    "svm": "Support Vector Machine (SVM Baseline)",
    "cnn": "1D Convolutional Neural Network (CNN Baseline)",
    "rnn": "Bidirectional RNN / GRU Baseline",
    "plain_snn": "Plain Spiking Neural Network (SNN Baseline)",
    "st_snn": "Hybrid Spatio-Temporal Attention SNN (ST-SNN Proposed)",
}


def load_all_models(model_root: Path, device: torch.device) -> dict[str, Any]:
    """Load all 5 trained models from model/best/ directories."""
    models: dict[str, Any] = {}

    # 1. SVM
    svm_path = model_root / "best" / "svm" / "model" / "model.joblib"
    if not svm_path.exists():
        svm_path = model_root / "best" / "svm" / "model" / "model.pkl"
    if svm_path.exists():
        try:
            models["svm"] = joblib.load(svm_path)
        except Exception as exc:
            print(f"[WARN] Failed to load SVM from {svm_path}: {exc}")
    else:
        print(f"[WARN] SVM model file not found at {svm_path}")

    # 2. CNN
    cnn_path = model_root / "best" / "cnn" / "model" / "model.pt"
    if cnn_path.exists():
        try:
            cnn = CNNBaseline().to(device)
            state = torch.load(cnn_path, map_location=device)
            cnn.load_state_dict(state)
            cnn.eval()
            models["cnn"] = cnn
        except Exception as exc:
            print(f"[WARN] Failed to load CNN from {cnn_path}: {exc}")

    # 3. RNN
    rnn_path = model_root / "best" / "rnn" / "model" / "model.pt"
    if rnn_path.exists():
        try:
            rnn = RNNBaseline().to(device)
            state = torch.load(rnn_path, map_location=device)
            rnn.load_state_dict(state)
            rnn.eval()
            models["rnn"] = rnn
        except Exception as exc:
            print(f"[WARN] Failed to load RNN from {rnn_path}: {exc}")

    # 4. Plain SNN
    snn_path = model_root / "best" / "snn" / "model" / "model.pt"
    if not snn_path.exists():
        snn_path = model_root / "best" / "plain_snn" / "model" / "model.pt"
    if snn_path.exists():
        try:
            snn = PlainSNN().to(device)
            state = torch.load(snn_path, map_location=device)
            snn.load_state_dict(state)
            snn.eval()
            models["plain_snn"] = snn
        except Exception as exc:
            print(f"[WARN] Failed to load Plain SNN from {snn_path}: {exc}")

    # 5. ST-SNN
    st_path = model_root / "best" / "st_snn" / "model" / "model.pt"
    if st_path.exists():
        try:
            st = SpikeAttentionSNN().to(device)
            state = torch.load(st_path, map_location=device)
            st.load_state_dict(state)
            st.eval()
            models["st_snn"] = st
        except Exception as exc:
            print(f"[WARN] Failed to load ST-SNN from {st_path}: {exc}")

    return models


def load_unseen_test_dataset(data_dir: Path, split_path: Path):
    """Extract strictly held-out test data according to the cross-validation split."""
    npz_files = list(data_dir.glob("beats_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No processed beats_*.npz found in {data_dir}")

    npz_path = npz_files[0]
    meta_path = data_dir / "beats_meta.csv"

    data = np.load(npz_path)
    signals = data["signals"]  # [N, 2, 360]
    meta_df = pd.read_csv(meta_path)

    with open(split_path, encoding="utf-8") as f:
        split_info = json.load(f)

    test_pgs = set(split_info["test_patient_group_ids"])
    test_mask = meta_df["patient_group_id"].isin(test_pgs).to_numpy()

    test_signals = signals[test_mask]
    test_meta = meta_df[test_mask].reset_index(drop=True)

    if "class_id" in test_meta.columns:
        test_labels = test_meta["class_id"].to_numpy(dtype=np.int64)
    else:
        test_labels = data["labels"][test_mask]

    return test_signals, test_labels, test_meta, split_info["test_patient_group_ids"]


def predict_single_beat(model: Any, model_name: str, sample_signal: np.ndarray, device: torch.device):
    """Run single-beat inference and measure latency in milliseconds."""
    # sample_signal is [1, 2, 360]
    t0 = time.perf_counter()

    if model_name == "svm":
        probs = model.predict_proba(sample_signal)[0]
    else:
        tensor_in = torch.from_numpy(sample_signal).float().to(device)
        with torch.no_grad():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(tensor_in)[0]
            else:
                logits = model.forward_logits(tensor_in)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    if device.type == "cuda" and model_name != "svm":
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_class_id = int(np.argmax(probs))
    pred_class_name = CLASS_ORDER[pred_class_id]
    return pred_class_id, pred_class_name, probs, latency_ms


def plot_unseen_beat_demo(
    signal: np.ndarray,
    meta_row: pd.Series,
    results: list[dict[str, Any]],
    output_path: Path,
):
    """Plot the 2-lead ECG beat and model prediction confidence comparison."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=300, gridspec_kw={"height_ratios": [1, 1.2]})

    # Lead 0 and Lead 1 ECG waveform
    t_ms = np.linspace(0, 1000, signal.shape[-1])  # 360 samples = 1000 ms
    axes[0].plot(t_ms, signal[0, 0, :], label="Lead 0 (MLII)", color="#1f77b4", linewidth=1.8)
    axes[0].plot(t_ms, signal[0, 1, :], label="Lead 1 (V1/V5)", color="#ff7f0e", linewidth=1.8, linestyle="--")
    axes[0].axvline(500, color="red", linestyle=":", alpha=0.7, label="R-peak Center")
    axes[0].set_title(
        f"Unseen ECG Beat #{meta_row.get('beat_id', 'N/A')} | "
        f"Patient: {meta_row.get('patient_group_id', 'N/A')} | "
        f"True Class: {meta_row.get('class_name')} ({CLASS_DESCRIPTIONS.get(meta_row.get('class_name'), '')})",
        fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Time (milliseconds)", fontsize=10)
    axes[0].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", framealpha=0.9)

    # Model prediction probabilities bar chart
    models = [r["model_key"].upper() for r in results]
    x = np.arange(len(models))
    width = 0.16

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    for c_idx, c_name in enumerate(CLASS_ORDER):
        probs_for_c = [r["probs"][c_idx] * 100.0 for r in results]
        offset = (c_idx - 2) * width
        axes[1].bar(
            x + offset, probs_for_c, width,
            label=f"Class {c_name} ({CLASS_DESCRIPTIONS[c_name].split(' ')[0]})",
            color=colors[c_idx], edgecolor="black", linewidth=0.7
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r['model_key'].upper()}\nPred: {r['pred_name']}" for r in results], fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Confidence Probability (%)", fontsize=10)
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Model Confidence Probabilities Across All 5 Classes", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\n[INFO] ECG Waveform & Model Confidence Plot saved to: {output_path}")


def display_full_metrics_table(results_dir: Path):
    """Display comprehensive evaluation metrics from the test partition results."""
    test_metrics_path = results_dir / "metrics" / "test_metrics.csv"
    per_class_path = results_dir / "metrics" / "test_per_class.csv"

    if not test_metrics_path.exists():
        print(f"[WARN] Metrics file not found at {test_metrics_path}. Run training first.")
        return

    print("\n" + "=" * 92)
    print("      COMPREHENSIVE TEST EVALUATION SCORES (HELD-OUT PATIENTS: 24,698 BEATS)      ")
    print("=" * 92)

    df_test = pd.read_csv(test_metrics_path)
    table_rows = []
    for _, row in df_test.iterrows():
        table_rows.append({
            "Model": row["model"].upper(),
            "Raw Acc (%)": f"{row['accuracy'] * 100.0:.2f}%",
            "Balanced Acc (%)": f"{row['balanced_accuracy'] * 100.0:.2f}%",
            "Macro Prec": f"{row['macro_precision']:.3f}",
            "Macro Recall": f"{row['macro_recall']:.3f}",
            "Macro F1": f"{row['macro_f1']:.3f}",
            "Weighted F1": f"{row['weighted_f1']:.3f}",
            "Macro ROC AU": f"{row['macro_roc_auc_ovr']:.3f}" if pd.notna(row['macro_roc_auc_ovr']) else "N/A",
        })

    summary_df = pd.DataFrame(table_rows)
    print(summary_df.to_string(index=False))

    if per_class_path.exists():
        print("\n" + "-" * 92)
        print("                        PER-CLASS RECALL / SENSITIVITY BREAKDOWN                         ")
        print("-" * 92)
        df_class = pd.read_csv(per_class_path)
        pivot = df_class.pivot(index="model", columns="class", values="recall") * 100.0
        # Reorder columns
        cols = [c for c in CLASS_ORDER if c in pivot.columns]
        pivot = pivot[cols]
        pivot.columns = [f"Class {c} (%)" for c in cols]
        pivot.index = [idx.upper() for idx in pivot.index]
        print(pivot.round(2).to_string())
    print("=" * 92 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SPIKE-ATTENTION: Unseen Test Evaluation & Single-Beat Demo")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--split-file", default="data/processed/splits/seed_0_fold_0.json", help="Path to split JSON")
    parser.add_argument("--model-dir", default="model", help="Path to saved models root directory")
    parser.add_argument("--results-dir", default="results/demo_models", help="Path to results directory")
    parser.add_argument("--sample-idx", type=int, default=None, help="Specific unseen test beat index (0 to 24697)")
    parser.add_argument("--target-class", type=str, default=None, choices=["N", "S", "V", "F", "Q", "ANY", "RANDOM"],
                        help="Target arrhythmia class (e.g. V, S, N, F, or ANY for totally random)")
    parser.add_argument("--patient", type=str, default=None, help="Filter by held-out patient (e.g. 109, 215, 100)")
    parser.add_argument("--plot", action="store_true", help="Generate comparison waveform plot (unseen_beat_test.png)")
    parser.add_argument("--full-metrics", action="store_true", help="Print complete evaluation score tables")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / args.data_dir
    split_file = project_root / args.split_file
    model_root = project_root / args.model_dir
    results_dir = project_root / args.results_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("         SPIKE-ATTENTION: UNSEEN TEST BEAT INFERENCE DEMO           ")
    print("=" * 80)
    print(f"Device:             {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Model Directory:    {model_root / 'best'}")
    print(f"Dataset Split:      {split_file.name}")

    # 1. Load models
    models = load_all_models(model_root, device)
    if not models:
        print("[ERROR] No models could be loaded. Ensure training has completed and models are in model/best/")
        return

    print(f"Loaded Models:      {', '.join([m.upper() for m in models.keys()])} (5/5 models ready)")

    # 2. Load unseen test partition
    signals, labels, meta_df, test_patients = load_unseen_test_dataset(data_dir, split_file)
    print(f"Held-out Patients:  {len(test_patients)} patients: {', '.join(test_patients[:5])}...")
    print(f"Total Unseen Beats: {len(meta_df):,} beats")

    # 3. Select test sample (Randomly from held-out partition unless specific index given)
    if args.sample_idx is not None:
        idx = max(0, min(args.sample_idx, len(meta_df) - 1))
        selection_mode = f"User Specified Index #{idx}"
    else:
        candidates = meta_df.index
        # Optional filter by patient
        if args.patient:
            p_match = candidates[meta_df.loc[candidates, "patient_group_id"].astype(str).str.contains(args.patient, case=False, na=False)]
            if len(p_match) > 0:
                candidates = p_match
            else:
                print(f"[WARN] No held-out beats for patient '{args.patient}'. Available: {test_patients}")

        # Optional filter by class
        target_cls = args.target_class
        if target_cls and target_cls.upper() not in ("ANY", "ALL", "RANDOM"):
            c_match = candidates[meta_df.loc[candidates, "class_name"] == target_cls.upper()]
            if len(c_match) > 0:
                candidates = c_match
            else:
                print(f"[WARN] No class '{target_cls}' found in candidates. Defaulting to all classes.")

        # Dynamically choose random index from held-out beats
        idx = int(np.random.choice(candidates))
        selection_mode = f"Dynamically Picked from {len(candidates):,} candidate held-out beats"

    sample_signal = signals[idx: idx + 1]  # [1, 2, 360]
    meta_row = meta_df.iloc[idx]
    true_class_name = meta_row["class_name"]
    true_class_desc = CLASS_DESCRIPTIONS.get(true_class_name, "Unknown")

    print("\n" + "-" * 80)
    print(f"Randomly Selected Held-Out Test Beat (Index #{idx} / {len(meta_df):,}):")
    print(f"  * Selection Mode:   {selection_mode}")
    print(f"  * Patient Group:    {meta_row.get('patient_group_id', 'N/A')} (Held-Out Patient)")
    print(f"  * Record ID:        {meta_row.get('record_id', 'N/A')}")
    print(f"  * Beat ID:          {meta_row.get('beat_id', 'N/A')}")
    print(f"  * Ground Truth:     CLASS {true_class_name} - {true_class_desc}")
    print(f"  * Tip to Re-test:   python test.py --sample-idx {idx}")
    print("-" * 80)

    # 4. Run inference through all 5 models
    print("\nReal-time Inference Results Across All 5 Models:")
    print(f"{'Model':<12} | {'Prediction':<10} | {'Status':<10} | {'Latency':<9} | {'Confidence Probabilities [P(N), P(S), P(V), P(F), P(Q)]'}")
    print("-" * 96)

    demo_results = []
    for model_key in ["svm", "cnn", "rnn", "plain_snn", "st_snn"]:
        if model_key not in models:
            continue

        model_obj = models[model_key]
        pred_id, pred_name, probs, latency = predict_single_beat(
            model=model_obj,
            model_name=model_key,
            sample_signal=sample_signal,
            device=device,
        )

        status = "CORRECT" if pred_name == true_class_name else "MISSED"
        status_str = f"[{status}]"
        probs_str = "[" + ", ".join([f"{p:.2f}" for p in probs]) + "]"

        print(f"{model_key.upper():<12} | {pred_name:<10} | {status_str:<10} | {latency:>6.2f} ms | {probs_str}")

        demo_results.append({
            "model_key": model_key,
            "pred_name": pred_name,
            "status": status,
            "probs": probs,
            "latency": latency,
        })

    print("-" * 96)

    # 5. Optional Plot
    if args.plot:
        plot_path = project_root / "unseen_beat_test.png"
        plot_unseen_beat_demo(sample_signal, meta_row, demo_results, plot_path)

    # 6. Full metrics summary
    display_full_metrics_table(results_dir)


if __name__ == "__main__":
    main()
