# Fair Model Benchmark — Sections 3.1.1.2 & 3.1.2.1

**Date:** 2026-06-27   **Test cell:** B0018 (held out, never trained on)
**Scripts:** `benchmark_models.py` (discharge), `benchmark_charge.py` (charge)
**Outputs:** `benchmark_discharge_parity.png`, `benchmark_charge_parity.png`, `*_metrics.csv`, `*.json`

## Fairness protocol (identical for every model)

| Control | Value |
|---|---|
| Train cells | B0005, B0006, B0007 |
| Test cell | B0018 (hold-out) |
| Label | Semin's `soh_pct` = capacity_Ah / 2.0 × 100 — **one shared source for all models** |
| Test cycles | the **same** B0018 cycles for every model (intersection of cycles each representation can produce) |
| Metrics | RMSE %, MAE %, R², 3-class accuracy (Good > 80 / Marginal 70–80 / Replace < 70) |
| Training budget | identical and modest for all neural models (fairness over per-model tuning) |
| Input | each model keeps its **author's designed representation** (we compare representations, not handicap them) |

> The relative ranking and the discharge→charge behaviour are the robust, decision-relevant
> findings. Absolute numbers come from a quick unified training budget and will move a little
> once each owner does final tuning — but they must keep this identical protocol.

---

## 3.1.1.2 — Discharge-curve benchmark (all 5 models)

504 train cycles / 132 B0018 test cycles.

Both train and test (B0018 hold-out) metrics are shown — the train↔test gap is the
overfitting / generalization diagnostic. (Full MAE per split is in `benchmark_discharge_metrics.csv`.)

| Model | Member | Input representation | Train R² | Train RMSE% | **Test R²** | **Test RMSE%** | Test acc3 |
|---|---|---|---:|---:|---:|---:|---:|
| **1D-CNN** | Evan | (4,128) V / \|I\| / T / cumQ waveform | 0.969 | 1.82 | **0.997** | **0.39** | 0.962 |
| **PINN** | Donghyun | 4 SOC-window feats (cap_ratio, dv_norm) | 0.997 | 0.60 | **0.995** | **0.56** | 0.955 |
| **PI-1D-CNN** | Evan | (4,128) waveform + physics head (Re, Rct) | 0.956 | 2.18 | **0.943** | **1.84** | 0.758 |
| SVM | Semin | hybrid scalar (Re/Rct + early V/I/T) | 0.997 | 0.61 | **0.484** | **5.54** | 0.644 |
| MLP | Evan | hybrid scalar (same as SVM) | −0.019 | 10.51 | **−0.001** | **7.72** | 0.356 |

**Top-3 (by test R²): 1D-CNN, PINN, PI-1D-CNN.**

Note the diagnostics: **SVM** trains to R²=0.997 but tests at 0.484 — a textbook overfit (it
memorizes the three training cells and fails to transfer to B0018). **MLP** fails to fit *or*
generalize (R²≈0 on both) with these scalar features. The capacity-based models (1D-CNN, PINN)
are the only ones with a small train↔test gap on discharge — for the leakage reason below.

**Key caveat — discharge capacity leakage.** On a discharge cycle, capacity ≈ SOH by
definition. Every top model consumes discharge capacity in some form — the CNN's cumulative-charge
channel (cumQ) and the PINN's `cap_ratio` feature (capacity over a fixed SOC window ≈ 0.2 × SOH).
That is *why* they reach R² ≈ 0.99, and why the shape-only scalar models (SVM, MLP) lag. So the
discharge result selects the strongest **architectures** but is **not** a fair test of true SOH
inference. The charge curve removes this leakage.

---

## 3.1.2.1 — Charging-curve benchmark (top-3) — the decisive, leakage-free test

489 train cycles / 130 B0018 test cycles. Charge label = SOH of the *following* discharge
(leakage-free). On charge, cumulative charge depends on starting SOC and the CV tail, so it does
**not** encode SOH — the model must learn curve shape.

| Model | Member | Input representation | Train R² | Train RMSE% | **Test R²** | **Test RMSE%** | Test acc3 |
|---|---|---|---:|---:|---:|---:|---:|
| **PI-1D-CNN** | Evan | (4,128) charge waveform + physics head | 0.974 | 1.63 | **0.814** | **3.29** | 0.708 |
| **1D-CNN** | Evan | (4,128) charge waveform | 0.982 | 1.36 | **0.810** | **3.32** | 0.723 |
| PINN | Donghyun | 4 charge V-window feats | 0.977 | 1.52 | **0.121** | **7.15** | 0.592 |

**Result.** The waveform CNNs **generalise** from the leaky discharge regime to the honest charge
regime (test R² ≈ 0.81 with a modest train↔test gap), while the scalar-capacity **PINN collapses**
(charge train R² 0.977 → **test 0.121**). That gap is the tell: the PINN's charge features memorize
the training cells but carry no transferable SOH signal once capacity no longer encodes SOH. The
waveform CNN, which reads the CC voltage trajectory and the CC→CV transition, keeps working.

---

## Decision: final model = **PI-1D-CNN** (1D-CNN is the lightweight runner-up)

It is essentially tied with the plain 1D-CNN on the charge test (0.814 vs 0.810) but adds an
interpretable physics head (recovers Re/Rct), which is valuable for flagging out-of-distribution
cells and for the Intel-deployment story. The plain 1D-CNN is the lightweight fallback. The scalar
SVM/MLP/PINN approaches are documented as baselines that do not survive the leakage-free test.

## For Donghyun (3.1.2.2 segment ±10% SOC)

Top-3 to carry into the segment study: **1D-CNN, PI-1D-CNN, PINN.** Amend `train_soh_segment.py`
to run these three under this same protocol (same train/test split, same labels, same metrics).
Per your note on fixed-load timing: at NASA's 2 A discharge, a ±10% SOC window of a ~2 Ah cell is
~0.2 Ah → ~6 min; on charge at 1.5 A CC it is ~8 min — both well inside the 30-minute demo slot,
which is the practicality argument for segment screening.

## Honesty notes (do not hide these in the report)

1. Numbers are from a **unified quick-training harness** for fairness, not each member's tuned best.
   Final report numbers should be re-run per model **under this same protocol**.
2. Charge waveforms come from Semin's 101-point CSV resampled to 128; discharge waveforms from the
   `.mat` files. Labels are identical across models.