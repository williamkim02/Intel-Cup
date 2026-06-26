# SVR Training by Input Type

This folder contains a GitHub-ready script for training and comparing SVM/SVR battery SOH models by input type.

## Model titles

- Model A — Basic Resistance SVR (Re/Rct Only)
- Model B — Early Discharge V/I/T SVR
- Model C — Hybrid Discharge SVR (Re/Rct + Early V/I/T)
- Model D — Early Charging Scalar SVR
- Model E — Full Charging Scalar SVR
- Model F — Charging + Impedance SVR
- Model G — Full Charging-Cycle Waveform PCA-SVR

## Classification thresholds

- Good: SOH > 80%
- Marginal: 70% <= SOH <= 80%
- Replace: SOH < 70%

The script reports both regression metrics and classification accuracy:
- MAE
- RMSE
- R2
- Classification accuracy
- Confusion matrix

## Expected data layout

Recommended GitHub layout:

```text
data/
  nasa_all_cells_discharge_features.csv
  nasa_all_cells_charge_features.csv
  nasa_all_cells_charge_waveform_101.csv

scripts/
  train_svr_by_input_with_accuracy.py

outputs/
```

The script also supports the original package paths:

```text
intel_dk2500_svr_package/data/nasa_all_cells_discharge_features.csv
intel_charge_cycle_work/data_processed/nasa_all_cells_charge_features.csv
intel_charge_cycle_work/data_processed/nasa_all_cells_charge_waveform_101.csv
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python scripts/train_svr_by_input_with_accuracy.py
```

Or with explicit paths:

```bash
python scripts/train_svr_by_input_with_accuracy.py \
  --discharge-csv data/nasa_all_cells_discharge_features.csv \
  --charge-csv data/nasa_all_cells_charge_features.csv \
  --charge-waveform-csv data/nasa_all_cells_charge_waveform_101.csv \
  --out-dir outputs/svr_training_by_input_results
```

## Outputs

```text
outputs/svr_training_by_input_results/
  svr_all_input_summary.csv
  svr_all_input_summary.json
  A_basic_resistance_svr/
    metrics.json
    b0018_predictions.csv
    A_basic_resistance_svr_b0018_soh_curve.png
    A_basic_resistance_svr_predicted_vs_true.png
    svr_model.joblib
  ...
```

## Notes for report

Train/test split:
- Train: B0005, B0006, B0007
- Test: B0018

Target:
- SOH = measured discharge capacity / 2.0 Ah * 100

For charging-cycle models:
- The input comes from the charge cycle.
- The SOH label is matched from the next corresponding discharge-cycle capacity.

Leakage control:
- Cycle number, cell ID, true capacity, true SOH, RUL, and normalized capacity are not used as model inputs.
