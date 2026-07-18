# -*- coding: utf-8 -*-
"""
postprocess_and_check.py — real DK-2500 measured discharge → deployed model

Self-contained, path-relative. Run from anywhere:
    python "Real Data/postprocess_and_check.py"

Pipeline:
  1. Read raw/discharge_usable_segment.csv (the valid CC-discharge phase).
  2. Denoise the voltage (±0.15 V raw noise) and convert to the dashboard schema
     (time, voltage, current, capacity_ah) → processed/*.csv.
  3. Run the DEPLOYED segment PINN (../models/soh_segment_model.pth — voltage-only
     [V_start, V_end, ΔV, SOC_mid], leakage-free) on a 10% sliding window and average.
  4. Compare with coulomb-counted SOH.  NOTHING here feeds capacity to the model.
"""
import os, sys, numpy as np, pandas as pd
from scipy.signal import medfilt, savgol_filter
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from pinn_model import SOHCurvePINN

RAW      = os.path.join(HERE, "raw", "discharge_usable_segment.csv")
OUTDIR   = os.path.join(HERE, "processed")
MODEL    = os.path.join(REPO, "models", "soh_segment_model.pth")
RATED_AH = 2.0                      # NASA 18650 rated; used ONLY for coulomb-SOH + Ah display
N_TS, WIN, STRIDE = 128, 13, 6      # 128-step resample, 10% window, 5% stride
os.makedirs(OUTDIR, exist_ok=True)

# ---- 1-2. denoise + convert to model-ready schema --------------------------
def denoise(v):
    v = medfilt(np.asarray(v, float), kernel_size=51)          # ~5 s median
    win = 101 if len(v) > 101 else (len(v) // 2 * 2 - 1)
    return savgol_filter(v, window_length=win, polyorder=2)

def build(sec, I, V, hz=1):
    t = sec - sec[0]
    keep = t >= 2.0                                            # drop 2 s settling
    t, I, V = t[keep] - t[keep][0], I[keep], denoise(V[keep])
    V = np.minimum.accumulate(V)                              # enforce discharge monotonicity
    out = pd.DataFrame({"time": t, "voltage": V, "current": np.abs(I)})
    dt = np.diff(out.time.values, prepend=out.time.values[0])
    out["capacity_ah"] = np.cumsum(out.current.values * dt) / 3600.0
    step = max(1, int(round(10 / hz)))
    return out.iloc[::step].reset_index(drop=True)

raw = pd.read_csv(RAW)
series = {"packmean": (raw.Vcell1 + raw.Vcell2) / 2,
          "cell1": raw.Vcell1, "cell2": raw.Vcell2}
outs = {}
for name, v in series.items():
    o = build(raw.sec.values, raw.Ipack.values, v.values)
    o.to_csv(os.path.join(OUTDIR, f"real_{name}_model_input.csv"), index=False)
    outs[name] = o
    print(f"processed/real_{name}_model_input.csv  rows={len(o)}  "
          f"V {o.voltage.iloc[0]:.3f}->{o.voltage.iloc[-1]:.3f}  cap={o.capacity_ah.iloc[-1]:.3f}Ah")

# ---- 3. deployed voltage-window segment PINN -------------------------------
ck   = torch.load(MODEL, weights_only=False, map_location="cpu")
arch = ck["arch"]; assert ck["features"] == ["V_start", "V_end", "dV", "SOC_mid"], ck["features"]
model = SOHCurvePINN(arch["n_features"], arch["hidden_dim"], arch["hidden_layers"])
model.load_state_dict(ck["model"]); model.eval(); sc = ck["scaler"]

def soh_windows(V):
    Vr = np.interp(np.linspace(0, 1, N_TS), np.linspace(0, 1, len(V)), V)
    preds = []
    for ws in range(0, N_TS - WIN + 1, STRIDE):
        we = ws + WIN
        v_s, v_e = float(Vr[ws]), float(Vr[we - 1])
        x = np.array([[v_s, v_e, v_s - v_e, 1.0 - (ws + we) / 2 / N_TS]], np.float32)
        with torch.no_grad():
            preds.append(float(model(torch.tensor(sc.transform(x).astype(np.float32)))) * 100)
    return np.array(preds)

def cls3(y): return "Good(>80)" if y > 80 else ("Marginal(70-80)" if y >= 70 else "Replace(<70)")

print("\n==== deployed segment PINN (voltage-only, no capacity) vs coulomb ====")
rows = []
for name, o in outs.items():
    p = soh_windows(o.voltage.values)
    meas = o.capacity_ah.iloc[-1] / RATED_AH * 100
    print(f"  {name:9s}  model={p.mean():5.1f}%  ({cls3(p.mean())})   coulomb={meas:5.1f}%   "
          f"|gap|={abs(p.mean()-meas):.1f}p   per-window {p.min():.0f}-{p.max():.0f}")
    rows.append(dict(source=name, segment_SOH=round(p.mean(), 1),
                     measured_coulomb_SOH=round(meas, 1),
                     gap_pp=round(abs(p.mean() - meas), 1), n_windows=len(p)))
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "soh_summary.csv"), index=False)
print("\nwrote processed/soh_summary.csv")
