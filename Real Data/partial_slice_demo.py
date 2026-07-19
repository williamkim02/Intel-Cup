# -*- coding: utf-8 -*-
"""
partial_slice_demo.py — score SOH from only a SHORT raw slice (no full discharge).

Takes just a few minutes of the raw DK-2500 discharge (e.g. any 5 min ≈ 8% SOC),
denoises that chunk, reads [V_start, V_end, ΔV] and estimates SOC_mid from voltage
(OCV→SOC), and runs the deployed segment PINN. No full curve, no capacity, no rated
value — this is the real second-life screening scenario.

Result: any 5-min slice → ~93% SOH vs coulomb truth 95.9% (full 56.6-min discharge).
"""
import os, sys, numpy as np, pandas as pd, torch
from scipy.signal import medfilt, savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from pinn_model import SOHCurvePINN
from ocv_soc import voltage_to_soc

ck = torch.load(os.path.join(REPO, "models", "soh_segment_model.pth"),
                weights_only=False, map_location="cpu")
m = SOHCurvePINN(4, 128, 3); m.load_state_dict(ck["model"]); m.eval(); sc = ck["scaler"]
def predict(vs, ve, smid):
    x = sc.transform(np.array([[vs, ve, vs - ve, smid]], np.float32)).astype(np.float32)
    with torch.no_grad():
        return float(m(torch.tensor(x))) * 100

raw = pd.read_csv(os.path.join(HERE, "raw", "discharge_usable_segment.csv"))
t = (raw.sec.values - raw.sec.values[0]) / 60.0
V = ((raw.Vcell1 + raw.Vcell2) / 2).values
full_min, COULOMB, FULLAVG = t.max(), 95.9, 94.0

def score_slice(start, dur):
    msk = (t >= start) & (t < start + dur)
    cd = medfilt(V[msk], 51)
    w = 101 if len(cd) > 101 else len(cd) // 2 * 2 - 1
    cd = np.minimum.accumulate(savgol_filter(cd, w, 2))
    vs, ve = float(cd[0]), float(cd[-1])
    smid = float(voltage_to_soc((vs + ve) / 2))
    return dict(start=start, dur=dur, vs=vs, ve=ve, smid=smid, soh=predict(vs, ve, smid))

slices = [score_slice(*s) for s in [(8,5),(20,5),(35,5),(20,10),(30,10)]]
print(f"FULL discharge {full_min:.1f} min | coulomb truth {COULOMB}% | full-curve avg {FULLAVG}%")
for s in slices:
    print(f"  t={s['start']:>2}-{s['start']+s['dur']:>2} min ({s['dur']:>2} min, "
          f"{s['dur']/full_min*100:>3.0f}% of full)  ->  SOH {s['soh']:.1f}%  "
          f"[V {s['vs']:.3f}->{s['ve']:.3f}, SOC_mid {s['smid']:.2f}]")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
ax[0].plot(t, V, color="#1a6fe0", lw=1.4, alpha=.5, label="full 56.6-min discharge")
for c, s in zip(["#f5a623", "#e0562f", "#8e44ad"], slices[:3]):
    msk = (t >= s["start"]) & (t < s["start"] + s["dur"])
    ax[0].plot(t[msk], V[msk], color=c, lw=3.5, label=f"{s['start']}–{s['start']+s['dur']} min → {s['soh']:.0f}%")
ax[0].set_xlabel("Time (min)"); ax[0].set_ylabel("Voltage (V)")
ax[0].set_title("Use only a 5-min slice (≈8% SOC)\nnot the full discharge"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

lbl = [f"{s['start']}-{s['start']+s['dur']}min\n({s['dur']}min)" for s in slices]
soh = [s["soh"] for s in slices]
cols = ["#f5a623","#e0562f","#8e44ad","#2f8fb0","#2f8fb0"]
ax[1].bar(lbl, soh, color=cols, edgecolor="white")
for i, v in enumerate(soh): ax[1].text(i, v+0.15, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
ax[1].axhline(COULOMB, ls="--", color="#39c66b", label=f"coulomb truth {COULOMB}%")
ax[1].axhline(FULLAVG, ls=":", color="#555", label=f"full-curve avg {FULLAVG}%")
ax[1].set_ylim(88, 98); ax[1].set_ylabel("SOH from the slice (%)")
ax[1].set_title("SOH from a short slice only\n(no full curve, no capacity, no rated)")
ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)

plt.suptitle("Partial-slice screening: a few minutes of raw discharge → SOH — deployed segment PINN",
             fontsize=12.5, fontweight="bold")
plt.tight_layout()
out = os.path.join(HERE, "figures", "partial_slice_demo.png")
plt.savefig(out, dpi=140); print("saved", out)
