#!/usr/bin/env python3
"""
Train SVM/SVR battery SOH models by input type and report accuracy.

Project:
    AI-Based Health Screening for Second-Life Lithium-Ion Batteries

Purpose:
    This script compares several Support Vector Regression (SVR) input designs
    for battery SOH estimation using NASA battery cells.

Train/test split:
    Train: B0005 + B0006 + B0007
    Test : B0018

Target:
    SOH (%) = measured discharge capacity / 2.0 Ah * 100

Health classification thresholds:
    Good     : SOH > 80%
    Marginal : 70% <= SOH <= 80%
    Replace  : SOH < 70%

Models trained:
    Model A — Basic Resistance SVR (Re/Rct Only)
    Model B — Early Discharge V/I/T SVR
    Model C — Hybrid Discharge SVR (Re/Rct + Early V/I/T)
    Model D — Early Charging Scalar SVR
    Model E — Full Charging Scalar SVR
    Model F — Charging + Impedance SVR
    Model G — Full Charging-Cycle Waveform PCA-SVR

Outputs:
    outputs/svr_training_by_input_results/
        svr_all_input_summary.csv
        svr_all_input_summary.json
        <model_folder>/
            metrics.json
            b0018_predictions.csv
            *_b0018_soh_curve.png
            *_predicted_vs_true.png
            trained .joblib model

Example:
    python scripts/train_svr_by_input_with_accuracy.py \
        --discharge-csv data/nasa_all_cells_discharge_features.csv \
        --charge-csv data/nasa_all_cells_charge_features.csv \
        --charge-waveform-csv data/nasa_all_cells_charge_waveform_101.csv \
        --out-dir outputs/svr_training_by_input_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
)


TRAIN_CELLS = ["B0005", "B0006", "B0007"]
TEST_CELL = "B0018"


# =============================================================================
# Feature definitions
# =============================================================================

DISCHARGE_FEATURE_SETS: Dict[str, Dict] = {
    "A_basic_resistance_svr": {
        "title": "Model A — Basic Resistance SVR (Re/Rct Only)",
        "input_description": "Re_ohm and Rct_ohm only",
        "features": ["Re_ohm", "Rct_ohm"],
        "cycle_col": "discharge_cycle",
        "C": 100.0,
    },
    "B_early_discharge_vit_svr": {
        "title": "Model B — Early Discharge V/I/T SVR",
        "input_description": "Early voltage, current, and temperature discharge features",
        "features": [
            "V_start", "V_10s", "V_30s", "V_60s",
            "V_drop_10s", "V_drop_30s", "V_drop_60s",
            "dV_dt_avg", "I_abs_avg",
            "T_start", "T_60s", "T_rise_60s",
        ],
        "cycle_col": "discharge_cycle",
        "C": 100.0,
    },
    "C_hybrid_discharge_svr": {
        "title": "Model C — Hybrid Discharge SVR (Re/Rct + Early V/I/T)",
        "input_description": "Re_ohm, Rct_ohm, and early voltage/current/temperature discharge features",
        "features": [
            "Re_ohm", "Rct_ohm",
            "V_start", "V_10s", "V_30s", "V_60s",
            "V_drop_10s", "V_drop_30s", "V_drop_60s",
            "dV_dt_avg", "I_abs_avg",
            "T_start", "T_60s", "T_rise_60s",
        ],
        "cycle_col": "discharge_cycle",
        "C": 100.0,
    },
}


EARLY_CHARGE_FEATURES = [
    "V_initial_rest", "V_5s", "V_10s", "V_30s", "V_60s", "V_120s", "V_300s",
    "V_rise_10s_from_5s", "V_rise_30s_from_5s", "V_rise_60s_from_5s",
    "V_rise_120s_from_5s", "V_rise_300s_from_5s",
    "dV_dt_5_60s", "dV_dt_60_300s",
    "I_5s", "I_10s", "I_30s", "I_60s", "I_120s", "I_300s",
    "I_avg_0_60s", "I_avg_0_300s",
    "T_start", "T_60s", "T_120s", "T_300s",
    "T_rise_60s", "T_rise_120s", "T_rise_300s",
    "cumQ_60s_Ah", "cumQ_120s_Ah", "cumQ_300s_Ah",
    "time_to_4p0V_s", "time_to_4p1V_s",
]

FULL_CHARGE_EXTRA_FEATURES = [
    "charge_time_s", "charge_Ah_calc", "V_end", "V_avg", "V_max",
    "I_end", "I_avg", "I_pos_avg", "I_avg_last_300s",
    "T_end", "T_avg", "T_max", "T_rise_total",
    "CC_duration_I_gt_1A_s", "CV_or_taper_duration_I_lt_0p1A_s",
    "time_to_4p15V_s", "time_to_4p2V_s",
    "Current_charge_avg", "Current_charge_end",
    "Voltage_charge_avg", "Voltage_charge_end",
]

CHARGE_FEATURE_SETS: Dict[str, Dict] = {
    "D_early_charging_scalar_svr": {
        "title": "Model D — Early Charging Scalar SVR",
        "input_description": "Early charging voltage, current, temperature, and cumulative charge features",
        "features": EARLY_CHARGE_FEATURES,
        "cycle_col": "charge_cycle",
        "C": 50.0,
    },
    "E_full_charging_scalar_svr": {
        "title": "Model E — Full Charging Scalar SVR",
        "input_description": "Early charging features plus full-charge summary features",
        "features": EARLY_CHARGE_FEATURES + FULL_CHARGE_EXTRA_FEATURES,
        "cycle_col": "charge_cycle",
        "C": 50.0,
    },
    "F_charging_plus_impedance_svr": {
        "title": "Model F — Charging + Impedance SVR",
        "input_description": "Full charging scalar features plus Re_ohm and Rct_ohm",
        "features": EARLY_CHARGE_FEATURES + FULL_CHARGE_EXTRA_FEATURES + ["Re_ohm", "Rct_ohm"],
        "cycle_col": "charge_cycle",
        "C": 50.0,
    },
}


# =============================================================================
# Utility functions
# =============================================================================

def resolve_existing_path(user_path: str, candidates: List[str]) -> Optional[Path]:
    """Resolve user path or fallback candidate paths."""
    paths = [Path(user_path)] + [Path(p) for p in candidates]
    for p in paths:
        if p.exists():
            return p
    return None


def classify_soh(values) -> np.ndarray:
    """Convert continuous SOH to 3-class health label."""
    y = np.asarray(values, dtype=float)
    return np.where(y > 80.0, "Good", np.where(y >= 70.0, "Marginal", "Replace"))


def make_svr_model(
    C: float,
    epsilon: float,
    use_pca: bool = False,
    pca_components: int = 20,
    n_train: int = 0,
    n_features: int = 0,
) -> Pipeline:
    """Create preprocessing + SVR pipeline."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if use_pca:
        n_comp = min(pca_components, max(1, n_train - 1), max(1, n_features))
        steps.append(("pca", PCA(n_components=n_comp, random_state=0)))

    steps.append(("svr", SVR(kernel="rbf", C=C, gamma="scale", epsilon=epsilon)))
    return Pipeline(steps)


def calculate_metrics(y_true, y_pred) -> Dict:
    """Calculate regression metrics and classification accuracy."""
    true_cls = classify_soh(y_true)
    pred_cls = classify_soh(y_pred)
    labels = ["Good", "Marginal", "Replace"]

    return {
        "mae_pct_points": float(mean_absolute_error(y_true, y_pred)),
        "rmse_pct_points": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "classification_accuracy": float(accuracy_score(true_cls, pred_cls)),
        "classification_accuracy_pct": float(accuracy_score(true_cls, pred_cls) * 100.0),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(true_cls, pred_cls, labels=labels).tolist(),
    }


def check_columns(df: pd.DataFrame, features: List[str], model_title: str) -> bool:
    """Check whether all required input features exist."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[SKIP] {model_title}: missing columns: {missing}")
        return False
    return True


def save_plots(
    pred_df: pd.DataFrame,
    metrics: Dict,
    title: str,
    cycle_col: str,
    out_dir: Path,
    safe_name: str,
) -> None:
    """Save SOH-vs-cycle and predicted-vs-true plots with metrics in the title."""
    y_true = pred_df["soh_pct"]
    y_pred = pred_df["predicted_soh_pct"]
    acc_pct = metrics["classification_accuracy_pct"]

    # SOH vs cycle plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(pred_df[cycle_col], y_true, "o-", markersize=3, linewidth=1.5, label="True SOH, B0018")
    plt.plot(pred_df[cycle_col], y_pred, "--", linewidth=2.0, label="SVR prediction")
    plt.axhline(80, linestyle=":", linewidth=1, label="Good/Marginal threshold")
    plt.axhline(70, linestyle=":", linewidth=1, label="Marginal/Replace threshold")
    plt.xlabel("Cycle")
    plt.ylabel("SOH (%)")
    plt.title(
        f"{title}: B0018 Verification\n"
        f"MAE={metrics['mae_pct_points']:.2f}%p, "
        f"RMSE={metrics['rmse_pct_points']:.2f}%p, "
        f"R²={metrics['r2']:.3f}, "
        f"Classification Accuracy={acc_pct:.2f}%"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{safe_name}_b0018_soh_curve.png", dpi=220)
    plt.close()

    # Predicted vs true plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=22, alpha=0.75)
    lo = min(float(np.min(y_true)), float(np.min(y_pred))) - 1
    hi = max(float(np.max(y_true)), float(np.max(y_pred))) + 1
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1.2, label="Ideal")
    plt.xlabel("True SOH (%)")
    plt.ylabel("Predicted SOH (%)")
    plt.title(
        f"{title}: Predicted vs True\n"
        f"MAE={metrics['mae_pct_points']:.2f}%p, "
        f"Classification Accuracy={acc_pct:.2f}%"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{safe_name}_predicted_vs_true.png", dpi=220)
    plt.close()


# =============================================================================
# Training functions
# =============================================================================

def train_scalar_svr(
    df: pd.DataFrame,
    model_key: str,
    config: Dict,
    out_dir: Path,
    epsilon: float,
) -> Optional[Dict]:
    """Train one scalar-feature SVR model."""
    title = config["title"]
    features = config["features"]
    cycle_col = config["cycle_col"]
    C = float(config["C"])

    if not check_columns(df, features, title):
        return None

    train = df[df["cell_id"].isin(TRAIN_CELLS)].copy()
    test = df[df["cell_id"] == TEST_CELL].copy()

    if train.empty or test.empty:
        print(f"[SKIP] {title}: train or test split is empty.")
        return None

    X_train = train[features]
    y_train = train["soh_pct"]
    X_test = test[features]
    y_test = test["soh_pct"]

    model = make_svr_model(C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, pred)
    metrics.update({
        "model_key": model_key,
        "model_title": title,
        "input_description": config["input_description"],
        "feature_names": features,
        "n_features": int(len(features)),
        "train_cells": TRAIN_CELLS,
        "test_cell": TEST_CELL,
        "n_train_samples": int(len(train)),
        "n_test_samples": int(len(test)),
        "svr_kernel": "rbf",
        "svr_C": C,
        "svr_epsilon": epsilon,
        "thresholds": {"Good": ">80", "Marginal": "70-80", "Replace": "<70"},
        "leakage_note": (
            "No cycle number, cell_id, true capacity, true SOH, RUL, or normalized capacity "
            "is used as a model input."
        ),
    })

    model_dir = out_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in ["cell_id", cycle_col, "soh_pct", "health_label"] if c in test.columns]
    pred_df = test[cols].copy()
    if cycle_col not in pred_df.columns:
        pred_df[cycle_col] = np.arange(1, len(pred_df) + 1)
    pred_df["predicted_soh_pct"] = pred
    pred_df["predicted_health_label"] = classify_soh(pred)
    pred_df["abs_error_pct_points"] = np.abs(pred_df["soh_pct"] - pred_df["predicted_soh_pct"])

    pred_df.to_csv(model_dir / "b0018_predictions.csv", index=False)
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, model_dir / "svr_model.joblib")

    save_plots(pred_df, metrics, title, cycle_col, model_dir, model_key)

    print_model_result(metrics)
    return metrics


def train_waveform_svr(
    waveform_csv: Path,
    out_dir: Path,
    C: float,
    epsilon: float,
    pca_components: int,
) -> Optional[Dict]:
    """Train full charging-cycle waveform PCA-SVR."""
    if not waveform_csv.exists():
        print(f"[SKIP] Model G — Full Charging-Cycle Waveform PCA-SVR: file not found: {waveform_csv}")
        return None

    df = pd.read_csv(waveform_csv)

    features = [c for c in df.columns if c.startswith(("V_", "I_", "T_", "Q_"))]
    features += [c for c in ["charge_time_s", "charge_Ah"] if c in df.columns]

    title = "Model G — Full Charging-Cycle Waveform PCA-SVR"
    if not check_columns(df, features, title):
        return None

    train = df[df["cell_id"].isin(TRAIN_CELLS)].copy()
    test = df[df["cell_id"] == TEST_CELL].copy()

    if train.empty or test.empty:
        print(f"[SKIP] {title}: train or test split is empty.")
        return None

    X_train = train[features]
    y_train = train["soh_pct"]
    X_test = test[features]
    y_test = test["soh_pct"]

    model = make_svr_model(
        C=C,
        epsilon=epsilon,
        use_pca=True,
        pca_components=pca_components,
        n_train=len(train),
        n_features=len(features),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    pca_step = model.named_steps.get("pca")
    n_comp = int(pca_step.n_components_) if pca_step is not None else None

    metrics = calculate_metrics(y_test, pred)
    metrics.update({
        "model_key": "G_full_charging_waveform_pca_svr",
        "model_title": title,
        "input_description": (
            "Full charging waveform: resampled V(t), I(t), T(t), cumQ(t), "
            "plus charge_time_s and charge_Ah."
        ),
        "feature_names": features,
        "n_features": int(len(features)),
        "pca_components": n_comp,
        "train_cells": TRAIN_CELLS,
        "test_cell": TEST_CELL,
        "n_train_samples": int(len(train)),
        "n_test_samples": int(len(test)),
        "svr_kernel": "rbf",
        "svr_C": C,
        "svr_epsilon": epsilon,
        "thresholds": {"Good": ">80", "Marginal": "70-80", "Replace": "<70"},
        "label_note": "SOH label is matched from the next corresponding discharge-cycle capacity.",
        "leakage_note": (
            "No cycle number, cell_id, true capacity, true SOH, RUL, or normalized capacity "
            "is used as a model input. charge_Ah is included because it is measured during "
            "the controlled charging process."
        ),
    })

    model_dir = out_dir / "G_full_charging_waveform_pca_svr"
    model_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in ["cell_id", "charge_cycle", "matched_discharge_cycle", "soh_pct", "health_label"] if c in test.columns]
    pred_df = test[cols].copy()
    if "charge_cycle" not in pred_df.columns:
        pred_df["charge_cycle"] = np.arange(1, len(pred_df) + 1)

    pred_df["predicted_soh_pct"] = pred
    pred_df["predicted_health_label"] = classify_soh(pred)
    pred_df["abs_error_pct_points"] = np.abs(pred_df["soh_pct"] - pred_df["predicted_soh_pct"])

    pred_df.to_csv(model_dir / "b0018_predictions.csv", index=False)
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, model_dir / "svr_pca_model.joblib")

    save_plots(
        pred_df=pred_df,
        metrics=metrics,
        title=title,
        cycle_col="charge_cycle",
        out_dir=model_dir,
        safe_name="G_full_charging_waveform_pca_svr",
    )

    print_model_result(metrics)
    return metrics


def print_model_result(metrics: Dict) -> None:
    """Pretty terminal output."""
    print("\n" + metrics["model_title"])
    print("-" * len(metrics["model_title"]))
    print(f"Input: {metrics['input_description']}")
    print(f"Train samples: {metrics['n_train_samples']}, Test samples: {metrics['n_test_samples']}")
    print(f"MAE: {metrics['mae_pct_points']:.2f} percentage points")
    print(f"RMSE: {metrics['rmse_pct_points']:.2f} percentage points")
    print(f"R2: {metrics['r2']:.3f}")
    print(f"Classification accuracy: {metrics['classification_accuracy_pct']:.2f}%")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discharge-csv", default="data/nasa_all_cells_discharge_features.csv")
    parser.add_argument("--charge-csv", default="data/nasa_all_cells_charge_features.csv")
    parser.add_argument("--charge-waveform-csv", default="data/nasa_all_cells_charge_waveform_101.csv")
    parser.add_argument("--out-dir", default="outputs/svr_training_by_input_results")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--waveform-C", type=float, default=100.0)
    parser.add_argument("--pca-components", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict] = []

    # Support both clean GitHub data/ layout and original package layouts.
    discharge_csv = resolve_existing_path(
        args.discharge_csv,
        [
            "intel_dk2500_svr_package/data/nasa_all_cells_discharge_features.csv",
            "data_processed/nasa_all_cells_discharge_features.csv",
        ],
    )
    charge_csv = resolve_existing_path(
        args.charge_csv,
        [
            "intel_charge_cycle_work/data_processed/nasa_all_cells_charge_features.csv",
            "data_processed/nasa_all_cells_charge_features.csv",
        ],
    )
    charge_waveform_csv = resolve_existing_path(
        args.charge_waveform_csv,
        [
            "intel_charge_cycle_work/data_processed/nasa_all_cells_charge_waveform_101.csv",
            "data_processed/nasa_all_cells_charge_waveform_101.csv",
        ],
    )

    if discharge_csv is not None:
        print(f"Using discharge CSV: {discharge_csv}")
        discharge_df = pd.read_csv(discharge_csv)
        for key, config in DISCHARGE_FEATURE_SETS.items():
            result = train_scalar_svr(discharge_df, key, config, out_dir, epsilon=args.epsilon)
            if result is not None:
                all_metrics.append(result)
    else:
        print("[INFO] Discharge CSV not found. Skipping discharge SVR models A-C.")

    if charge_csv is not None:
        print(f"\nUsing charge scalar CSV: {charge_csv}")
        charge_df = pd.read_csv(charge_csv)
        for key, config in CHARGE_FEATURE_SETS.items():
            result = train_scalar_svr(charge_df, key, config, out_dir, epsilon=args.epsilon)
            if result is not None:
                all_metrics.append(result)
    else:
        print("[INFO] Charge scalar CSV not found. Skipping charge scalar SVR models D-F.")

    if charge_waveform_csv is not None:
        print(f"\nUsing charge waveform CSV: {charge_waveform_csv}")
        result = train_waveform_svr(
            waveform_csv=charge_waveform_csv,
            out_dir=out_dir,
            C=args.waveform_C,
            epsilon=args.epsilon,
            pca_components=args.pca_components,
        )
        if result is not None:
            all_metrics.append(result)
    else:
        print("[INFO] Charge waveform CSV not found. Skipping waveform SVR model G.")

    if not all_metrics:
        raise RuntimeError("No models were trained. Please check the input CSV paths.")

    summary = pd.DataFrame([
        {
            "model_key": m["model_key"],
            "model_title": m["model_title"],
            "input_description": m["input_description"],
            "n_features": m["n_features"],
            "n_train_samples": m["n_train_samples"],
            "n_test_samples": m["n_test_samples"],
            "MAE_%p": m["mae_pct_points"],
            "RMSE_%p": m["rmse_pct_points"],
            "R2": m["r2"],
            "classification_accuracy_%": m["classification_accuracy_pct"],
        }
        for m in all_metrics
    ])

    summary_csv = out_dir / "svr_all_input_summary.csv"
    summary_json = out_dir / "svr_all_input_summary.json"
    summary.to_csv(summary_csv, index=False)

    with open(summary_json, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n==============================")
    print("SVR INPUT COMPARISON SUMMARY")
    print("==============================")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(summary_csv)
    print(summary_json)
    print("\nEach model folder contains metrics.json, predictions.csv, plots, and a trained .joblib model.")


if __name__ == "__main__":
    main()
