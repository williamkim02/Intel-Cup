"""
train_segment.py — 10% partial-discharge SOH prediction (sliding window)

Approach:
  - Slide a 10% window (WIN=13/128 steps) with 5% stride over each 128-step discharge waveform
  - Each window → one SOH prediction
  - Any 10% segment of the discharge curve is sufficient to predict SOH

Models:
  - Segment-PINN  : nf=4 [V_start, V_end, dV, soc_mid]
  - Segment-PI-1D-CNN : nc=3 [V, |I|, T] window resampled to 128 steps

LOCO-CV on B0005/B0006/B0007 for hyperparameter selection (B0018 never seen).
Final evaluation: per-segment R² AND per-cycle averaged R².
"""
import os, copy, warnings
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore"); np.random.seed(42); torch.manual_seed(42)
ROOT     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models"); os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_CELLS = ["B0005","B0006","B0007"]; TEST_CELL = "B0018"
N_TS = 128; WIN = 13; STRIDE = 6   # 13/128 ≈ 10%, stride 6/128 ≈ 5%

# ── Helpers ──────────────────────────────────────────────────────────────────
def cls3(y): y=np.asarray(y,float); return np.where(y>80,0,np.where(y>=70,1,2))
def metrics(yt, yp):
    yt,yp = np.asarray(yt,float), np.asarray(yp,float)
    return dict(RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                MAE =float(mean_absolute_error(yt,yp)),
                R2  =float(r2_score(yt,yp)),
                ACC3=float(accuracy_score(cls3(yt),cls3(yp))))

def extract_segments(wavs, sohs, cells, res, rcts, is_tests):
    """Slide 10% window over each 128-step discharge waveform."""
    rows = []
    for i in range(len(wavs)):
        V = wavs[i, 0]   # voltage channel (128,)
        for ws in range(0, N_TS - WIN + 1, STRIDE):
            we = ws + WIN
            # soc_mid: 1.0 = start of discharge, ~0.0 = end
            soc_mid = 1.0 - (ws + we) / 2.0 / N_TS
            v_s = float(V[ws]); v_e = float(V[we-1]); dv = v_s - v_e
            pinn_f = np.array([v_s, v_e, dv, soc_mid], np.float32)
            # resample 3×WIN → 3×128
            seg = wavs[i, :, ws:we]
            cnn_w = np.stack([
                np.interp(np.linspace(0,1,N_TS), np.linspace(0,1,WIN), seg[c])
                for c in range(3)
            ]).astype(np.float32)
            rows.append(dict(
                cell=str(cells[i]), soh=float(sohs[i]),
                pinn=pinn_f, wav=cnn_w,
                Re=float(res[i]), Rct=float(rcts[i]),
                is_test=bool(is_tests[i]),
                cycle_idx=int(i)   # original cycle index for per-cycle aggregation
            ))
    return pd.DataFrame(rows)

# ── Models ───────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(s, nc=3, drop=0.3):
        super().__init__()
        s.feat = nn.Sequential(
            nn.Conv1d(nc,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(drop), nn.AdaptiveAvgPool1d(1))
        s.reg = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32,1))
        s.phy = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,2))
    def forward(s, x):
        f = s.feat(x).squeeze(-1)
        return torch.sigmoid(s.reg(f)), s.phy(f)

class PINN(nn.Module):
    def __init__(s, nf=4, h=128, L=3):
        super().__init__()
        layers = [nn.Linear(nf,h), nn.Tanh()]
        for _ in range(L-1): layers += [nn.Linear(h,h), nn.Tanh()]
        layers += [nn.Linear(h,1)]
        s.net = nn.Sequential(*layers)
    def forward(s, x): return torch.sigmoid(s.net(x))

def seg_pinn_loss(model, x, y, lb=0.1):
    """MSE + boundary loss (SOH > 0.5). No monotonicity: not well-defined for segment features."""
    p = model(x)
    return ((p-y)**2).mean() + lb*torch.relu(0.5-p).pow(2).mean()

# ── Training functions ────────────────────────────────────────────────────────
def train_cnn(Wtr_n, ytr_t, phytr, Wte_n, drop=0.1, lr=1e-3, wd=1e-4, ep_max=150, seed=42):
    torch.manual_seed(seed)
    net = CNN(nc=3, drop=drop)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n = Wtr_n.shape[0]; idx = np.arange(n)
    for ep in range(ep_max):
        net.train(); np.random.seed(ep); np.random.shuffle(idx)
        for b in range(0, n, 64):
            bi = idx[b:b+64]; soh, phy = net(Wtr_n[bi])
            loss = ((soh - ytr_t[bi])**2).mean() + 0.01*((phy - phytr[bi])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        yp_tr = net(Wtr_n)[0].numpy().ravel() * 100
        yp_te = net(Wte_n)[0].numpy().ravel() * 100
    return net, yp_tr, yp_te

def train_pinn(Xtr, ytr_raw, Xev, Xte, h=128, L=3, lb=0.1, ep=3000, seed=42):
    nstd = np.array([0.005, 0.005, 0.002, 0.010], np.float32)
    rng  = np.random.default_rng(seed)
    Xa=[Xtr]; Ya=[ytr_raw]
    for _ in range(20):
        Xa.append(Xtr + rng.normal(0,1,Xtr.shape).astype(np.float32)*nstd)
        Ya.append(ytr_raw)
    sc    = MinMaxScaler().fit(np.vstack(Xa))
    Xtr_n = torch.tensor(sc.transform(np.vstack(Xa)).astype(np.float32))
    ytr_n = torch.tensor(np.concatenate(Ya)).view(-1,1)
    Xev_n = torch.tensor(sc.transform(Xev).astype(np.float32))
    Xte_n = torch.tensor(sc.transform(Xte).astype(np.float32))
    torch.manual_seed(seed)
    model = PINN(nf=4, h=h, L=L)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch   = ReduceLROnPlateau(opt, patience=150, factor=0.5, min_lr=1e-5)
    best=1e9; best_sd=None
    for e in range(ep):
        model.train(); opt.zero_grad()
        loss = seg_pinn_loss(model, Xtr_n, ytr_n, lb=lb)
        loss.backward(); opt.step(); sch.step(loss.detach())
        if loss.item() < best: best=loss.item(); best_sd=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        yp_tr = model(Xev_n).numpy().ravel() * 100
        yp_te = model(Xte_n).numpy().ravel() * 100
    return model, sc, yp_tr, yp_te

# ── LOCO-CV ───────────────────────────────────────────────────────────────────
def loco_cv_cnn(df_tr):
    cells = TRAIN_CELLS; best_r2=-np.inf; best_cfg=None
    for drop in [0.1, 0.3]:
        for lr in [1e-3, 3e-3]:
            for wd in [1e-4, 1e-3]:
                cv_r2s = []
                for val_cell in cells:
                    tr_c = df_tr[df_tr.cell!=val_cell].reset_index(drop=True)
                    va_c = df_tr[df_tr.cell==val_cell].reset_index(drop=True)
                    Wt = torch.tensor(np.stack(tr_c.wav.values))
                    Wv = torch.tensor(np.stack(va_c.wav.values))
                    mu = Wt.mean(dim=(0,2),keepdim=True); sd = Wt.std(dim=(0,2),keepdim=True)+1e-6
                    Wt_n, Wv_n = (Wt-mu)/sd, (Wv-mu)/sd
                    yt_t = torch.tensor((tr_c.soh.values/100.).astype(np.float32)).view(-1,1)
                    pt   = torch.tensor(tr_c[["Re","Rct"]].values.astype(np.float32))
                    _, _, yp_val = train_cnn(Wt_n, yt_t, pt, Wv_n, drop=drop, lr=lr, wd=wd, ep_max=100)
                    cv_r2s.append(r2_score(va_c.soh.values, yp_val))
                avg = np.mean(cv_r2s)
                print(f"  [PI-CNN-seg] drop={drop} lr={lr} wd={wd}  avg_R2={avg:.3f}")
                if avg > best_r2: best_r2=avg; best_cfg=dict(drop=drop,lr=lr,wd=wd,ep_max=150)
    print(f"  [PI-CNN-seg] BEST: {best_cfg}  CV_R2={best_r2:.3f}")
    return best_cfg

def loco_cv_pinn(df_tr):
    cells = TRAIN_CELLS; best_r2=-np.inf; best_cfg=None
    for h in [64, 128]:
        for L in [2, 3]:
            for lb in [0.05, 0.1, 0.2]:
                cv_r2s = []
                for val_cell in cells:
                    tr_c = df_tr[df_tr.cell!=val_cell].reset_index(drop=True)
                    va_c = df_tr[df_tr.cell==val_cell].reset_index(drop=True)
                    Xtr  = np.vstack(tr_c.pinn.values).astype(np.float32)
                    ytr  = (tr_c.soh.values/100.).astype(np.float32)
                    Xva  = np.vstack(va_c.pinn.values).astype(np.float32)
                    _, _, _, yp_val = train_pinn(Xtr, ytr, Xtr, Xva, h=h, L=L, lb=lb, ep=1500)
                    cv_r2s.append(r2_score(va_c.soh.values, yp_val))
                avg = np.mean(cv_r2s)
                print(f"  [PINN-seg] h={h} L={L} lb={lb}  avg_R2={avg:.3f}")
                if avg > best_r2: best_r2=avg; best_cfg=dict(h=h,L=L,lb=lb)
    print(f"  [PINN-seg] BEST: {best_cfg}  CV_R2={best_r2:.3f}")
    return best_cfg

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load discharge NPZ
    npz = np.load(os.path.join(ROOT, "data", "discharge_dataset.npz"), allow_pickle=True)
    wavs     = npz['wav']       # (636, 3, 128)
    sohs     = npz['soh']
    cells    = npz['cell']
    res      = npz['Re_mOhm']   # mΩ
    rcts     = npz['Rct_mOhm']  # mΩ
    is_tests = npz['is_test']

    # Extract all segments
    print("Extracting 10% sliding window segments...")
    df = extract_segments(wavs, sohs, cells, res, rcts, is_tests)
    n_wins = (N_TS - WIN) // STRIDE + 1
    print(f"Segments: total={len(df)}  wins/cycle={n_wins}")
    print(f"  train={( ~df.is_test).sum()}  test={df.is_test.sum()}")

    tr = df[~df.is_test].reset_index(drop=True)
    te = df[ df.is_test].reset_index(drop=True)
    yt = te.soh.values; results={}; preds={}

    # Normalize CNN waveforms
    Wtr = torch.tensor(np.stack(tr.wav.values))
    Wte = torch.tensor(np.stack(te.wav.values))
    cmu = Wtr.mean(dim=(0,2), keepdim=True)
    csd = Wtr.std(dim=(0,2),  keepdim=True) + 1e-6
    Wtr_n, Wte_n = (Wtr-cmu)/csd, (Wte-cmu)/csd
    ytr_t  = torch.tensor((tr.soh.values/100.).astype(np.float32)).view(-1,1)
    phytr  = torch.tensor(tr[["Re","Rct"]].values.astype(np.float32))

    # ── LOCO-CV ──────────────────────────────────────────────────────────────
    print("\n=== PI-1D-CNN Segment LOCO-CV ===")
    cnn_cfg = loco_cv_cnn(tr)

    print("\n=== PINN Segment LOCO-CV ===")
    pinn_cfg = loco_cv_pinn(tr)

    # ── Final training ────────────────────────────────────────────────────────
    print("\n=== Final training ===")

    # PI-1D-CNN
    net_pi, yp_tr_pi, yp_te_pi = train_cnn(Wtr_n, ytr_t, phytr, Wte_n, **cnn_cfg)
    results["PI-1D-CNN-seg"] = {"train": metrics(tr.soh.values, yp_tr_pi), "test": metrics(yt, yp_te_pi)}
    preds["PI-1D-CNN-seg"]   = yp_te_pi
    torch.save({"state_dict": net_pi.state_dict(), "mu": cmu, "sd": csd, "cfg": cnn_cfg},
               os.path.join(MODEL_DIR, "pi_1dcnn_segment.pt"))

    # PINN
    Xtr     = np.vstack(tr.pinn.values).astype(np.float32)
    ytr_raw = (tr.soh.values/100.).astype(np.float32)
    Xte     = np.vstack(te.pinn.values).astype(np.float32)
    pinn_m, sc, yp_tr_pinn, yp_te_pinn = train_pinn(Xtr, ytr_raw, Xtr, Xte, **pinn_cfg, ep=3000)
    results["PINN-seg"] = {"train": metrics(tr.soh.values, yp_tr_pinn), "test": metrics(yt, yp_te_pinn)}
    preds["PINN-seg"]   = yp_te_pinn
    torch.save({"state_dict": pinn_m.state_dict(), "scaler": sc, "cfg": pinn_cfg},
               os.path.join(MODEL_DIR, "pinn_segment.pt"))

    # ── Per-segment results ───────────────────────────────────────────────────
    order = sorted(results, key=lambda k: results[k]["test"]["R2"], reverse=True)
    print("\n=== SEGMENT BENCHMARK (10% window, B0018 test) ===")
    print(f"{'Model':<18}{'split':>6}{'RMSE%':>8}{'MAE%':>8}{'R2':>9}{'Acc3':>8}")
    tbl = []
    for k in order:
        rt, rte = results[k]["train"], results[k]["test"]
        print(f"{k:<18}{'train':>6}{rt['RMSE']:>8.2f}{rt['MAE']:>8.2f}{rt['R2']:>9.3f}{rt['ACC3']:>8.3f}")
        print(f"{'':<18}{'test':>6}{rte['RMSE']:>8.2f}{rte['MAE']:>8.2f}{rte['R2']:>9.3f}{rte['ACC3']:>8.3f}")
        tbl.append(dict(model=k, train_R2=round(rt['R2'],4), test_R2=round(rte['R2'],4),
                        test_RMSE=round(rte['RMSE'],4), test_MAE=round(rte['MAE'],4)))

    # Per-cycle aggregation (mean of window predictions)
    print("\n=== PER-CYCLE aggregated (mean of windows) ===")
    for k, yp_seg in preds.items():
        te_copy = te.copy(); te_copy["yp"] = yp_seg
        cyc = te_copy.groupby("cycle_idx").agg(soh_true=("soh","first"), soh_pred=("yp","mean")).reset_index()
        r2c = r2_score(cyc.soh_true, cyc.soh_pred)
        rmse_c = np.sqrt(mean_squared_error(cyc.soh_true, cyc.soh_pred))
        print(f"  {k:<18}  cycle-R2={r2c:.3f}  cycle-RMSE={rmse_c:.2f}%")
        tbl_entry = next(t for t in tbl if t["model"]==k)
        tbl_entry["cycle_R2"] = round(r2c, 4); tbl_entry["cycle_RMSE"] = round(rmse_c, 4)

    pd.DataFrame(tbl).to_csv(os.path.join(ROOT, "metrics_segment.csv"), index=False)

    # ── Parity plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"PI-1D-CNN-seg": "steelblue", "PINN-seg": "darkorange"}

    # Per-segment scatter
    ax = axes[0]
    lim = [yt.min()-2, yt.max()+2]; ax.plot(lim, lim, "k--", lw=1, label="y=x")
    for k in order:
        r = results[k]["test"]
        ax.scatter(yt, preds[k], s=5, alpha=0.4, color=colors[k],
                   label=f"{k}: R²={r['R2']:.3f}")
    ax.set_xlabel("True SOH (%)"); ax.set_ylabel("Predicted SOH (%)")
    ax.set_title("Per-segment (10% window)\nB0018 hold-out")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Per-cycle scatter
    ax = axes[1]
    ax.plot(lim, lim, "k--", lw=1, label="y=x")
    for k, yp_seg in preds.items():
        te_copy = te.copy(); te_copy["yp"] = yp_seg
        cyc = te_copy.groupby("cycle_idx").agg(soh_true=("soh","first"), soh_pred=("yp","mean")).reset_index()
        r2c = r2_score(cyc.soh_true, cyc.soh_pred)
        ax.scatter(cyc.soh_true, cyc.soh_pred, s=30, alpha=0.7, color=colors[k],
                   label=f"{k}: R²={r2c:.3f}")
    ax.set_xlabel("True SOH (%)"); ax.set_ylabel("Predicted SOH (%)")
    ax.set_title("Per-cycle (mean of windows)\nB0018 hold-out")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("10% Partial Discharge SOH Prediction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "parity_segment.png"), dpi=130)
    print(f"\nSaved: metrics_segment.csv, parity_segment.png")
    print(f"Winner: {order[0]}")

if __name__ == "__main__": main()
