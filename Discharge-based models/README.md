# Discharge-based SOH models

Trained on **B0005/B0006/B0007**, evaluated on **B0018** (hold-out). Label = soh_pct
(capacity_Ah / 2.0 x 100). See `metrics_discharge.csv` and `parity_discharge.png`.

NOTE: on a discharge cycle capacity ~= SOH, so capacity-based inputs (CNN cumQ, PINN
cap_ratio) score very high partly by leakage. The charge-based models are the fair test.

## models/
| file | model | input |
|---|---|---|
| 1dcnn_discharge.pt    | 1D-CNN (PyTorch)     | (4,128) V/absI/T/cumQ waveform |
| pi_1dcnn_discharge.pt | PI-1D-CNN (PyTorch)  | (4,128) waveform + physics head (Re,Rct) |
| pinn_discharge.pt     | PINN (PyTorch)       | 4 SOC-window feats (cap_ratio, dv_norm) |
| svm_discharge.joblib  | SVR (scikit-learn)   | discharge-curve scalar (V/I/T only, NO impedance R) |
| mlp_discharge.joblib  | MLP (scikit-learn)   | same curve-only scalar |

Each .pt stores `state_dict` + norm stats (CNNs: `chan_mu/chan_sd`; PINN: `scaler_min/scaler_range`).
PINN uses Donghyun's recipe (MinMaxScaler, L = MSE + 0.5*L_mono + 0.1*L_bound, x30 noise, 3000 ep).
Rebuild the class from `train_discharge.py`, load_state_dict, normalize inputs, then infer.

## data/
- `nasa_all_cells_discharge_features.csv` — scalar features + Re/Rct + soh_pct (Semin's pipeline)
- `discharge_dataset.npz` — processed arrays: wav (N,4,128), scal, pinn, soh, cell, is_test, Re/Rct_mOhm
- raw `B0005/6/7/18.mat` (NASA PCoE) are needed only to regenerate the npz via `train_discharge.py`.

## reproduce
    python train_discharge.py        # retrains all 5, writes metrics + parity plot
