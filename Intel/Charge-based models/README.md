# Charge-based SOH models (top-3 from the discharge benchmark)

Trained on **B0005/B0006/B0007**, evaluated on **B0018** (hold-out). Charge label = SOH of
the *following* discharge cycle (leakage-free). See `metrics_charge.csv` / `parity_charge.png`.

This is the FAIR/decisive comparison: on charge, cumulative charge does not encode SOH, so the
model must learn curve shape. The waveform CNNs generalize (test R2 ~ 0.81); the scalar PINN
collapses (test R2 ~ 0.12).

## models/
| file | model | input |
|---|---|---|
| 1dcnn_charge.pt    | 1D-CNN (PyTorch)    | (4,128) V/absI/T/Q charge waveform |
| pi_1dcnn_charge.pt | PI-1D-CNN (PyTorch) | (4,128) waveform + physics head (Re,Rct) |
| pinn_charge.pt     | PINN (PyTorch)      | 4 charge V-window feats (cap, dV/dQ) |

Each .pt stores `state_dict` + normalization stats. Rebuild the class from `train_charge.py`.

## data/
- `nasa_all_cells_charge_waveform_101.csv` — 101-pt V/I/T/Q charge waveform + soh_pct
- `nasa_all_cells_charge_features.csv` — charge scalar features + Re/Rct
- `charge_dataset.npz` — processed arrays: wav (N,4,128), pinn, soh, cell, is_test, Re/Rct_mOhm

## reproduce

    python train_charge.py           # retrains the top-3, writes metrics + parity plot
