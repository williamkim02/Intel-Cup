# Real Data — DK-2500 measured discharge vs the deployed model

A real discharge measured on the **Intel DK-2500** rig with an **Imperix** power-electronics
logger (`battery_logger`, 17 Jul 2026), post-processed and run through the **deployed segment
PINN** — an independent, on-hardware check of the model from the final report.

## Layout
```
Real Data/
├── postprocess_and_check.py   # self-contained: raw → processed → deployed model → comparison
├── raw/
│   ├── data_log.csv               # original Imperix export (instrument header + samples)
│   ├── data_log_CLEANED.csv       # timestamp, sec, Ipack, Vcell1, Vcell2, phase (0.1 s)
│   └── discharge_usable_segment.csv   # the valid_discharge phase (used for SOH)
├── processed/                     # model-ready (dashboard schema: time, voltage, current, capacity_ah)
│   ├── real_packmean_model_input.csv  # ← upload this to the dashboard
│   ├── real_cell1_model_input.csv
│   ├── real_cell2_model_input.csv
│   └── soh_summary.csv
└── figures/
    ├── real_comparison.png        # discharge curve · per-window SOH · model-vs-measured
    └── denoise_before_after.png   # raw ±0.15 V noise → denoised model-ready voltage
```

## Measured discharge (usable segment)
- Constant-current **2.035 A** (matches NASA 2 A CC), **56.6 min**, **1.920 Ah** discharged.
- Cell voltage 4.11 → 3.0 V (partial depth); cell imbalance 60 mV mean.
- Raw voltage is ~±0.15 V noisy → denoised (5 s median → Savitzky-Golay) and downsampled to 1 Hz.

## Deployed model vs measured (rated 2.0 Ah)
The deployed segment PINN uses **voltage-only** window features `[V_start, V_end, ΔV, SOC_mid]` —
**no capacity, no rated capacity, no label** from this cell. It was trained only on NASA
B0005/B0006/B0007 and never saw this cell or any real-hardware data.

| Source | Segment PINN (model) | Coulomb-counted | Gap |
|--------|---------------------:|----------------:|----:|
| pack-mean | **94.0 %** | 95.9 % | 1.9 %p |
| cell 1 | 93.5 % | 95.9 % | 2.4 %p |
| cell 2 | 93.8 % | 95.9 % | 2.1 %p |

Same **Good** class, ~2 %p from an independent coulomb-counting measurement. Because the model
input carries no capacity, this agreement does **not** depend on the rated-capacity assumption —
only the coulomb "truth" side does.

## Honest scope
- This is one **healthy** cell (~94–96 %). The screening decision boundary (70–80 %) is **not** tested here.
- Per-window spread is real (79–99 %); the per-cycle mean is the reported metric — averaging windows is what stabilises it.
- Coulomb "truth" assumes the cell's rated capacity is 2.0 Ah; confirm the nameplate for an exact error.

## Reproduce
```bash
python "Real Data/postprocess_and_check.py"     # writes processed/*.csv + soh_summary.csv
```
Or use the dashboard: `streamlit run app.py` → *SOH — Quick Segment* → Upload CSV →
`Real Data/processed/real_packmean_model_input.csv`.
