# AI-Based Health Screening for Second-Life Lithium-Ion Batteries

**2026 Intel Cup Undergraduate Electronic Design Contest — Embedded System Design Invitational**
Nanyang Technological University · Kim Donghyun, Lee Hyunseung, Ban Semin

On-device **State-of-Health (SOH)** screening for retired EV lithium-ion cells, running entirely on the
**Intel DK-2500** edge platform through **OpenVINO** (CPU / NPU / iGPU, no host PC or cloud in the inference path).

The deployed model is a **physics-informed neural network (PINN)** that reads a single **10 % discharge-voltage
window** and predicts SOH on an unseen cell quickly and seed-robustly (**per-window R² = 0.895 ± 0.013**,
per-cycle ≈ 0.92). Results are shown in an interactive **Streamlit dashboard** with a Good / Marginal / Replace
decision plus knee-point and remaining-useful-life views.

> Dataset: NASA PCoE Li-ion Battery Aging (cells B0005/B0006/B0007 train, **B0018 held out**, leave-one-cell-out).

---

## Quick start

```bash
pip install -r requirements.txt        # numpy, pandas, scipy, scikit-learn, matplotlib, joblib, torch
streamlit run app.py                   # launch the dashboard  (Windows: run.bat)
```

The dashboard loads the trained weights in `models/` and screens a discharge log (CSV upload or manual entry).

---

## Repository structure

```
Intel-Cup/
├── app.py                     # Streamlit decision-support dashboard (deployed instrument)
├── pinn_model.py              # PINN model definition (shared by app + training)
├── knee_detector.py           # knee-point / lifetime-tracking logic (dashboard aux tab)
├── train_soh_segment.py       # trains the DEPLOYED ±10% segment PINN
├── train_soh_universal.py     # trains the full-discharge (universal) PINN
├── export_models.py           # PyTorch → ONNX → OpenVINO IR (FP16) export for DK-2500
├── benchmark_models.py        # fair discharge benchmark (SVR/MLP/1D-CNN/PI-1D-CNN/PINN)
├── benchmark_charge.py        # fair charge benchmark (leakage-free)
├── benchmark_*.{json,csv,png} # benchmark outputs (read by export_models.py)
├── MODEL_BENCHMARK_RESULTS.md # benchmark protocol + results summary
├── requirements.txt · run.bat
│
├── models/                    # trained weights loaded by app.py
│   ├── soh_segment_model.pth      # ★ deployed model (±10% segment PINN)
│   ├── soh_universal_model.pth    # full-discharge PINN
│   ├── rul_model.pth              # remaining-useful-life
│   ├── parking_model.pkl          # cold-weather / parking risk
│   └── SVR_SVM/                   # SVR/SVM baselines (resistance & curve features)
│
├── Discharge-based models/    # discharge + ±10% segment study (report §3.4.2 / §3.4.4)
└── Charge-based models/       # charge-curve study, leakage-free (report §3.4.3)
```

### How folders map to the report

| Folder / file | Report section | Role |
|---|---|---|
| `app.py`, `knee_detector.py`, `models/` | Ch.4 (End-to-End System) | Deployed dashboard + weights |
| `train_soh_segment.py` + `models/soh_segment_model.pth` | §3.4.4, §3.5 | **Deployed** ±10% segment PINN |
| `train_soh_universal.py` + `models/soh_universal_model.pth` | §3.4.2 | Full-discharge PINN reference |
| `Discharge-based models/` | §3.4.2, §3.4.4 | Discharge & segment representations |
| `Charge-based models/` | §3.4.3 | Charge representation (leakage-free) |
| `models/SVR_SVM/` | §3.3.1, §3.4.1 | SVR/MLP resistance & curve baselines |
| `benchmark_*`, `MODEL_BENCHMARK_RESULTS.md` | §3.4 | Fair cross-model comparison |
| `export_models.py` | §4.1 | OpenVINO FP16 deployment export |

---

## Models at a glance

| Model | Representation | Held-out B0018 | Notes |
|---|---|---|---|
| **Segment PINN** ★ | 10% discharge window (4 voltage feats) | **R² = 0.895 ± 0.013** (per-cycle ≈ 0.92) | deployed; fast + seed-robust + leakage-free |
| Full-discharge PINN | capacity-normalised window feats | R² = 0.944 ± 0.005 | needs whole curve; capacity-informed |
| Charge PINN | ICA (dV/dQ) + impedance | R² = 0.32 ± 0.27 | promising but seed-unstable, not deployed |
| SVR / MLP | resistance / curve scalars | R² < 0 (resistance only) | baselines; impedance alone doesn't transfer |
| 1D-CNN / PI-1D-CNN | raw waveform | collapses once capacity-leakage removed | diagnostic baseline |

---

## Reproduce

```bash
python benchmark_models.py      # discharge benchmark → benchmark_discharge_*.{csv,json,png}
python benchmark_charge.py      # charge benchmark   → benchmark_charge_*.{csv,json,png}
python train_soh_segment.py     # retrain deployed segment PINN
python export_models.py         # export to OpenVINO IR (FP16) for the DK-2500
```

Dataset: NASA Prognostics Center of Excellence, *Li-ion Battery Aging Datasets* (18650 cells, 2.0 Ah).

---

## Deployment (Intel DK-2500)

Intel Core Ultra 5 225U (12-core CPU + AI Boost NPU + iGPU), Ubuntu 22.04, OpenVINO 2026.
The FP16 segment PINN (~79 KB) runs sub-millisecond on all three engines; inference is assigned to the
**NPU** (7.1 W) to keep the continuously-running instrument within its 15 W envelope.
