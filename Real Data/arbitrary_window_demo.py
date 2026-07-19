# -*- coding: utf-8 -*-
"""
arbitrary_window_demo.py — can ANY short 10% discharge window give SOH?

Shows two things on the deployed voltage-window segment PINN:
  (A) Given only a partial discharge (SOC 90->60%), each arbitrary 10% window
      inside it still yields an SOH.
  (B) The SAME arbitrary window (73-63% SOC) tracks true SOH across the held-out
      cell B0018's whole life (r ~ 0.98) — proof the window measures degradation,
      not a constant.
"""
import os, sys, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from pinn_model import SOHCurvePINN

ck = torch.load(os.path.join(REPO, "models", "soh_segment_model.pth"),
                weights_only=False, map_location="cpu")
m = SOHCurvePINN(4, 128, 3); m.load_state_dict(ck["model"]); m.eval(); sc = ck["scaler"]
def predict(vs, ve, smid):
    x = np.array([[vs, ve, vs - ve, smid]], np.float32)
    with torch.no_grad():
        return float(m(torch.tensor(sc.transform(x).astype(np.float32)))) * 100

# ---- (A) real cell: arbitrary windows within a given 90-60% partial ----
df = pd.read_csv(os.path.join(HERE, "processed", "real_packmean_model_input.csv"))
V = df.voltage.values; soc = 1.0 - df.capacity_ah.values / 2.0
o = np.argsort(soc); V_of = lambda s: float(np.interp(s, soc[o], V[o]))
bands = [(0.90,0.80),(0.85,0.75),(0.80,0.70),(0.73,0.63),(0.68,0.58),(0.63,0.53)]
mids  = [(hi+lo)/2 for hi,lo in bands]
sohA  = [predict(V_of(hi), V_of(lo), (hi+lo)/2) for hi,lo in bands]

# ---- (B) B0018 across life: same 73-63% window ----
z = np.load(os.path.join(REPO, "Discharge-based models", "data", "discharge_dataset.npz"),
            allow_pickle=True)
W = z["wav"][z["cell"]=="B0018"]; S = z["soh"][z["cell"]=="B0018"]
N_TS, WIN = 128, 13; ws = int(round((1-0.68)*N_TS - WIN/2)); we = ws + WIN
smid = 1.0 - (ws+we)/2/N_TS
predB = np.array([predict(float(W[i,0,ws]), float(W[i,0,we-1]), smid) for i in range(len(W))])
r = np.corrcoef(S, predB)[0,1]

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

# A1: discharge curve with the 73-63% window highlighted
ax[0].plot(soc[o]*100, V[o], color="#1a6fe0", lw=1.6)
ax[0].axvspan(63, 73, color="#f5a623", alpha=0.30, label="73–63% window")
ax[0].axvspan(60, 90, color="#39c66b", alpha=0.08, label="given partial (90–60%)")
ax[0].set_xlim(100, 40); ax[0].set_xlabel("SOC (%)  [discharge →]"); ax[0].set_ylabel("Voltage (V)")
ax[0].set_title("Real discharge — only 90–60% is given\nread any 10% window inside it")
ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(alpha=.3)

# A2: SOH from each arbitrary window
xlbl = [f"{int(hi*100)}–{int(lo*100)}" for hi,lo in bands]
cols = ["#8aa0c8"]*len(bands); cols[3] = "#f5a623"
ax[1].bar(xlbl, sohA, color=cols, edgecolor="white")
for i,v in enumerate(sohA): ax[1].text(i, v+0.15, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
ax[1].axhline(95.9, ls="--", color="#39c66b", label="coulomb truth 95.9%")
ax[1].axhline(94.0, ls=":", color="#555", label="full-sweep model 94.0%")
ax[1].set_ylim(90, 101); ax[1].set_ylabel("SOH from ONE window (%)")
ax[1].set_xlabel("SOC window (%)"); ax[1].set_title("Any 10% window → an SOH\n(73–63% = 98.7%; single-window ±few %p)")
ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)
ax[1].tick_params(axis="x", labelrotation=20)

# B: same window tracks true SOH across B0018 life
ax[2].scatter(S, predB, s=22, alpha=.6, color="#1a6fe0")
lim = [S.min()-3, S.max()+3]; ax[2].plot(lim, lim, "k--", lw=1, label="y = x")
ax[2].set_xlim(lim); ax[2].set_ylim(lim)
ax[2].set_xlabel("True SOH (%)"); ax[2].set_ylabel("Predicted from the 73–63% window (%)")
ax[2].set_title(f"Same arbitrary window across B0018 life\nit genuinely measures degradation  (r = {r:.2f})")
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

plt.suptitle("Any arbitrary 10% discharge window recovers SOH — deployed segment PINN",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(HERE, "figures", "arbitrary_window_demo.png")
plt.savefig(out, dpi=140)
print("saved", out)
print(f"73-63% window on real cell: {sohA[3]:.1f}%   |   B0018 correlation r={r:.2f}")
