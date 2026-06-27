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
| Training recipe | each model uses its **own** formulas & epochs (PINN = Donghyun's exact `train_soh_universal.py`: MinMaxScaler, L = MSE + 0.5·L_mono + 0.1·L_bound, ×30 noise). Epochs need **not** be identical across models — only constraint is no excessive per-model finetuning |
| Input | each model's **author-designed representation**. SVM/MLP use **discharge-curve features only — impedance Re/Rct removed**. PI-CNN uses Re/Rct only as an auxiliary training *target*, never as an input |

> The relative ranking and the discharge→charge behaviour are the robust, decision-relevant
> findings. Numbers below are the **full 3000-epoch run** (PINN = Donghyun's exact recipe). The CNNs
> show run-to-run / platform variance on this small dataset (±~0.05 R²): the conclusions are stable,
> but the exact 1D-CNN vs PI-1D-CNN ordering on charge can flip between runs — see the Decision note.

---

## 3.1.1.2 — Discharge-curve benchmark (all 5 models)

504 train cycles / 132 B0018 test cycles.

Both train and test (B0018 hold-out) metrics are shown — the train↔test gap is the
overfitting / generalization diagnostic. (Full MAE per split is in `benchmark_discharge_metrics.csv`.)

| Model | Member | Input representation | Train R² | Train RMSE% | **Test R²** | **Test RMSE%** | Test acc3 |
|---|---|---|---:|---:|---:|---:|---:|
| **PINN** | Donghyun | 4 SOC-window feats (cap_ratio, dv_norm); L_mono+L_bound | 0.998 | 0.46 | **0.997** | **0.43** | 0.992 |
| **1D-CNN** | Evan | (4,128) V / \|I\| / T / cumQ waveform | 0.950 | 2.34 | **0.959** | **1.55** | 0.894 |
| **PI-1D-CNN** | Evan | (4,128) waveform + physics head (Re, Rct) | 0.942 | 2.51 | **0.808** | **3.38** | 0.629 |
| SVM | Semin | discharge-curve scalar (V/I/T only, **no R**) | 0.995 | 0.71 | **0.445** | **5.75** | 0.614 |
| MLP | Evan | same curve-only scalar as SVM | −0.068 | 10.76 | **−0.015** | **7.77** | 0.356 |

**Top-3 (by test R²): PINN, 1D-CNN, PI-1D-CNN.**

Note the diagnostics: **SVM** trains to R²=0.995 but tests at 0.445 — a textbook overfit (it
memorizes the three training cells and fails to transfer to B0018), and dropping the impedance R
inputs moved it only 0.484→0.445, confirming the curves, not R, carry the transferable signal.
**MLP** fails to fit *or* generalize (R²≈0 on both). The capacity-based models (1D-CNN, PINN) are
the only ones with a small train↔test gap on discharge — for the leakage reason below.

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
| **1D-CNN** | Evan | (4,128) charge waveform | 0.988 | 1.12 | **0.862** | **2.84** | 0.754 |
| **PI-1D-CNN** | Evan | (4,128) charge waveform + physics head | 0.952 | 2.21 | **0.717** | **4.06** | 0.692 |
| PINN | Donghyun | 4 charge V-window feats; L_mono+L_bound | 0.967 | 1.82 | **0.264** | **6.54** | 0.577 |

**Result.** The waveform CNNs **generalise** from the leaky discharge regime to the honest charge
regime (test R² 0.72–0.86), while the scalar-capacity **PINN collapses** (charge train R² 0.967 →
**test 0.264**). The tell is the train↔test gap: the PINN's charge features memorize the training
cells but carry no transferable SOH signal once capacity no longer encodes SOH — its monotonicity
prior is tuned for discharge cap_ratio and does not hold on charge. The waveform CNN, which reads
the CC voltage trajectory and the CC→CV transition, keeps working.

---

## Decision: the waveform-CNN family is the final model

On the leakage-free charge test, **1D-CNN (0.862)** and **PI-1D-CNN (0.717)** both clearly beat every
scalar model (PINN 0.264; SVM/MLP already fail on discharge). So the final model is a waveform 1D-CNN.
Between the two CNN variants the result varies run-to-run on this small dataset (±~0.05 R²): **this
full 3000-epoch run favours the plain 1D-CNN** (best charge accuracy, and the lightest model). The
**PI-1D-CNN** trades a little accuracy for an interpretable physics head (recovers Re/Rct) that helps
flag out-of-distribution cells and supports the Intel-deployment story. Recommendation: settle the
1D-CNN vs PI-1D-CNN choice with a **5-seed average** rather than one run — both are defensible. The
scalar SVM/MLP/PINN approaches are baselines that do not survive the leakage-free test. (Note: PINN
is the *best* model on discharge at 0.997 — but that is the leaky regime, which is exactly why the
charge test is the decider.)

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