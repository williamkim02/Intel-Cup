"""
export_models.py — train every benchmark model and consolidate the trained
artifacts + data into shareable folders for the GitHub repo.

Usage:
    python export_models.py discharge   # -> "Discharge-based models/"
    python export_models.py charge      # -> "Charge-based models/"

Each folder gets: models/ (saved weights), data/ (CSV + processed .npz),
the training script, the metrics CSV, the parity PNG, and a README.
"""
import os, sys, shutil, json
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

ROOT = os.path.dirname(os.path.abspath(__file__))
DISC_SVMDIR = os.path.join(ROOT, "Discharge-based models", "data")
CHRG_SVMDIR = os.path.join(ROOT, "Charge-based models", "data")
np.random.seed(42); torch.manual_seed(42)

def copy_if_new(src, dst_dir):
    """Copy src into dst_dir, skipping if it would copy a file onto itself
    (the export target and the canonical data source can be the same folder)."""
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst) and os.path.samefile(src, dst):
        return
    shutil.copy(src, dst)

# ---- model defs (identical to the benchmark scripts) ----
class CNN(nn.Module):
    def __init__(self, nc=4, physics=False):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(nc,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.reg = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
        self.physics = physics
        if physics: self.phy = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,2))
    def forward(self, x):
        f = self.feat(x).squeeze(-1); soh = torch.sigmoid(self.reg(f))
        return (soh, self.phy(f)) if self.physics else (soh, None)

class PINN(nn.Module):
    def __init__(self, nf=4, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),
                                 nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
    def forward(self, x): return torch.sigmoid(self.net(x))

def mkdirs(base):
    os.makedirs(os.path.join(base,"models"), exist_ok=True)
    os.makedirs(os.path.join(base,"data"), exist_ok=True)

def train_cnn(Wtr_n, ytr_w, phytr, physics, epochs):
    torch.manual_seed(42)
    net = CNN(4, physics); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    n = Wtr_n.shape[0]; idx = np.arange(n)
    for ep in range(epochs):
        net.train(); np.random.seed(ep); np.random.shuffle(idx)
        for b in range(0, n, 32):
            bi = idx[b:b+32]; soh, phy = net(Wtr_n[bi]); loss = ((soh-ytr_w[bi])**2).mean()
            if physics: loss = loss + 0.01*((phy-phytr[bi])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net

def train_pinn(Xraw, yraw):
    """Donghyun faithful recipe: MinMaxScaler, x30 noise, L=MSE+0.5*L_mono+0.1*L_bound, 3000 ep.
    Returns (trained net, fitted MinMaxScaler)."""
    Xraw = np.asarray(Xraw, np.float32); yraw = np.asarray(yraw, np.float32)
    nstd = np.array([0.0005,0.020,0.0005,0.020], np.float32); rng = np.random.default_rng(42)
    Xa=[Xraw]; Ya=[yraw]
    for _ in range(30): Xa.append(Xraw+rng.normal(0,1,Xraw.shape).astype(np.float32)*nstd); Ya.append(yraw)
    sc = MinMaxScaler().fit(np.vstack(Xa))
    Xt = torch.tensor(sc.transform(np.vstack(Xa)).astype(np.float32)); yt = torch.tensor(np.concatenate(Ya)).view(-1,1)
    def loss_fn(m,x,y,lm=0.5,lb=0.1):
        p=m(x); xp=x.detach().clone().requires_grad_(True)
        g=torch.autograd.grad(m(xp).sum(),xp,create_graph=True)[0]
        return ((p-y)**2).mean()+lm*torch.relu(-g).pow(2).mean()+lb*torch.relu(0.6-p).pow(2).mean()
    torch.manual_seed(42); net=PINN(); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    sch=ReduceLROnPlateau(opt,patience=200,factor=0.5,min_lr=1e-5); best=1e9; bs=None
    for ep in range(int(os.environ.get("PINN_EPOCHS","3000"))):
        net.train(); opt.zero_grad(); L=loss_fn(net,Xt,yt); L.backward(); opt.step(); sch.step(L.detach())
        if L.item()<best: best=L.item(); bs=copy.deepcopy(net.state_dict())
    net.load_state_dict(bs); net.eval(); return net, sc

# ==================================================================== DISCHARGE
def export_discharge():
    from benchmark_models import build_discharge_table, CURVE_FEATS
    base = os.path.join(ROOT, "Discharge-based models"); mkdirs(base)
    MD = os.path.join(base,"models"); DD = os.path.join(base,"data")
    df = build_discharge_table()
    tr = df[df.cell!="B0018"].reset_index(drop=True); te = df[df.cell=="B0018"].reset_index(drop=True)
    ytr = tr.soh.values

    # SVM + MLP (scalar)
    Xtr = np.vstack(tr.scal.values);
    svm = Pipeline([("sc",StandardScaler()),("svr",SVR(kernel="rbf",C=100,epsilon=0.1))]).fit(Xtr, ytr)
    joblib.dump({"model":svm,"features":CURVE_FEATS}, os.path.join(MD,"svm_discharge.joblib"))
    scaler = StandardScaler().fit(Xtr)
    mlp = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="tanh",max_iter=4000,
                       random_state=42,alpha=1e-3).fit(scaler.transform(Xtr), ytr)
    joblib.dump({"scaler":scaler,"model":mlp,"features":CURVE_FEATS}, os.path.join(MD,"mlp_discharge.joblib"))

    # PINN (4 SOC-window feats) — Donghyun faithful recipe
    pinn, psc = train_pinn(np.vstack(tr.pinn.values), ytr/100.)
    torch.save({"state_dict":pinn.state_dict(),
                "scaler_min":psc.data_min_.astype(np.float32),
                "scaler_range":(psc.data_max_-psc.data_min_).astype(np.float32),
                "features":["cap_ratio_100_80","dv_norm_100_80","cap_ratio_80_60","dv_norm_80_60"],
                "note":"x_norm = (x - scaler_min)/scaler_range  ; L = MSE + 0.5*L_mono + 0.1*L_bound"},
               os.path.join(MD,"pinn_discharge.pt"))

    # 1D-CNN + PI-1D-CNN (4,128 waveform)
    Wtr = torch.tensor(np.stack(tr.wav.values)); cmu = Wtr.mean(dim=(0,2),keepdim=True); csd = Wtr.std(dim=(0,2),keepdim=True)+1e-6
    Wtr_n = (Wtr-cmu)/csd; ytr_w = torch.tensor((ytr/100.).astype(np.float32)).view(-1,1)
    phytr = torch.tensor(tr[["Re","Rct"]].values.astype(np.float32))
    cnn = train_cnn(Wtr_n, ytr_w, phytr, False, 120)
    torch.save({"state_dict":cnn.state_dict(),"chan_mu":cmu,"chan_sd":csd,
                "channels":["V","absI","T","cumQ"]}, os.path.join(MD,"1dcnn_discharge.pt"))
    picnn = train_cnn(Wtr_n, ytr_w, phytr, True, 120)
    torch.save({"state_dict":picnn.state_dict(),"chan_mu":cmu,"chan_sd":csd,
                "channels":["V","absI","T","cumQ"],"physics_targets":["Re_mOhm","Rct_mOhm"]},
               os.path.join(MD,"pi_1dcnn_discharge.pt"))

    # processed dataset
    np.savez_compressed(os.path.join(DD,"discharge_dataset.npz"),
        wav=np.stack(df.wav.values), scal=np.vstack(df.scal.values), pinn=np.vstack(df.pinn.values),
        soh=df.soh.values, cell=df.cell.values, is_test=(df.cell=="B0018").values,
        Re_mOhm=df.Re.values, Rct_mOhm=df.Rct.values, scal_features=np.array(CURVE_FEATS))
    copy_if_new(os.path.join(DISC_SVMDIR,"nasa_all_cells_discharge_features.csv"), DD)
    shutil.copy(os.path.join(ROOT,"benchmark_models.py"), os.path.join(base,"train_discharge.py"))
    for f,dst in [("benchmark_discharge_metrics.csv","metrics_discharge.csv"),
                  ("benchmark_discharge_parity.png","parity_discharge.png")]:
        if os.path.exists(os.path.join(ROOT,f)): shutil.copy(os.path.join(ROOT,f), os.path.join(base,dst))
    open(os.path.join(base,"README.md"),"w").write(README_DISC)
    print("Wrote", base, "->", sorted(os.listdir(MD)))

# ==================================================================== CHARGE
def export_charge():
    from benchmark_charge import build_charge_table
    base = os.path.join(ROOT, "Charge-based models"); mkdirs(base)
    MD = os.path.join(base,"models"); DD = os.path.join(base,"data")
    df = build_charge_table()
    tr = df[df.cell!="B0018"].reset_index(drop=True); te = df[df.cell=="B0018"].reset_index(drop=True)
    ytr = tr.soh.values

    Wtr = torch.tensor(np.stack(tr.wav.values)); cmu = Wtr.mean(dim=(0,2),keepdim=True); csd = Wtr.std(dim=(0,2),keepdim=True)+1e-6
    Wtr_n = (Wtr-cmu)/csd; ytr_w = torch.tensor((ytr/100.).astype(np.float32)).view(-1,1)
    phytr = torch.tensor(tr[["Re","Rct"]].values.astype(np.float32))
    cnn = train_cnn(Wtr_n, ytr_w, phytr, False, 150)
    torch.save({"state_dict":cnn.state_dict(),"chan_mu":cmu,"chan_sd":csd,"channels":["V","absI","T","Q"]},
               os.path.join(MD,"1dcnn_charge.pt"))
    picnn = train_cnn(Wtr_n, ytr_w, phytr, True, 150)
    torch.save({"state_dict":picnn.state_dict(),"chan_mu":cmu,"chan_sd":csd,"channels":["V","absI","T","Q"],
                "physics_targets":["Re_mOhm","Rct_mOhm"]}, os.path.join(MD,"pi_1dcnn_charge.pt"))

    pinn, psc = train_pinn(np.vstack(tr.pinn.values), ytr/100.)
    torch.save({"state_dict":pinn.state_dict(),
                "scaler_min":psc.data_min_.astype(np.float32),
                "scaler_range":(psc.data_max_-psc.data_min_).astype(np.float32),
                "features":["cap_3.90_4.05","dvdq_3.90_4.05","cap_4.05_4.18","dvdq_4.05_4.18"],
                "note":"x_norm = (x - scaler_min)/scaler_range  ; L = MSE + 0.5*L_mono + 0.1*L_bound"},
               os.path.join(MD,"pinn_charge.pt"))

    np.savez_compressed(os.path.join(DD,"charge_dataset.npz"),
        wav=np.stack(df.wav.values), pinn=np.vstack(df.pinn.values), soh=df.soh.values,
        cell=df.cell.values, is_test=(df.cell=="B0018").values, Re_mOhm=df.Re.values, Rct_mOhm=df.Rct.values)
    for f in ["nasa_all_cells_charge_waveform_101.csv","nasa_all_cells_charge_features.csv"]:
        copy_if_new(os.path.join(CHRG_SVMDIR,f), DD)
    shutil.copy(os.path.join(ROOT,"benchmark_charge.py"), os.path.join(base,"train_charge.py"))
    for f,dst in [("benchmark_charge_metrics.csv","metrics_charge.csv"),
                  ("benchmark_charge_parity.png","parity_charge.png")]:
        if os.path.exists(os.path.join(ROOT,f)): shutil.copy(os.path.join(ROOT,f), os.path.join(base,dst))
    open(os.path.join(base,"README.md"),"w").write(README_CHRG)
    print("Wrote", base, "->", sorted(os.listdir(MD)))

README_DISC = """# Discharge-based SOH models

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
"""

README_CHRG = """# Charge-based SOH models (top-3 from the discharge benchmark)

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
"""

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("discharge","both"): export_discharge()
    if mode in ("charge","both"): export_charge()
