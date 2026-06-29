"""
train_charge.py — Charge-curve SOH benchmark (LOCO-CV clean version)

Changes from original:
- CNN: 3 channels (V, |I|, T) — Q removed (Q_final ≈ SOH×rated → leakage)
- PINN: ICA-style features (cap/RATED and dV/dQ in CC voltage windows) — not direct SOH leakage
- LOCO-CV on B0005/B0006/B0007 only for hyperparameter selection (B0018 never touched during tuning)
- Dropout added to CNN; models saved to models/
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
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR= os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_CELLS = ["B0005","B0006","B0007"]; TEST_CELL = "B0018"; N_TS = 128; RATED = 2.0

# ── Helpers ──────────────────────────────────────────────────────────────────
def cls3(y): y=np.asarray(y,float); return np.where(y>80,0,np.where(y>=70,1,2))
def metrics(yt, yp):
    yt,yp = np.asarray(yt,float), np.asarray(yp,float)
    return dict(RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                MAE =float(mean_absolute_error(yt,yp)),
                R2  =float(r2_score(yt,yp)),
                ACC3=float(accuracy_score(cls3(yt),cls3(yp))))
def resamp(x, n=N_TS):
    x = np.asarray(x, float)
    return np.interp(np.linspace(0,1,n), np.linspace(0,1,len(x)), x)

def charge_pinn_feats(V, Q):
    """ICA-style features: cap/RATED and dV/dQ in two CC voltage windows.
    Window capacity ≠ total discharge capacity, so NOT direct SOH leakage."""
    feats = []
    for lo, hi in [(3.90, 4.05), (4.05, 4.18)]:
        m = (V >= lo) & (V <= hi)
        if m.sum() < 3: return None
        q = Q[m]; v = V[m]; cap = q[-1] - q[0]
        if cap <= 1e-6: return None
        dvdq = (v[-1] - v[0]) / cap
        feats += [cap/RATED, dvdq*RATED]
    return np.array(feats, np.float32)

def build_charge_table():
    w  = pd.read_csv(os.path.join(DATA_DIR, "nasa_all_cells_charge_waveform_101.csv"))
    f  = pd.read_csv(os.path.join(DATA_DIR, "nasa_all_cells_charge_features.csv"))
    fk = f.set_index(["cell_id","charge_cycle"])[["Re_ohm","Rct_ohm"]]
    Vc=[f"V_{i:03d}" for i in range(101)]; Ic=[f"I_{i:03d}" for i in range(101)]
    Tc=[f"T_{i:03d}" for i in range(101)]; Qc=[f"Q_{i:03d}" for i in range(101)]
    rows = []
    for _, r in w.iterrows():
        cell = r.cell_id; cc = int(r.charge_cycle)
        if (cell, cc) not in fk.index: continue
        V=r[Vc].values.astype(float); I=r[Ic].values.astype(float)
        T=r[Tc].values.astype(float); Q=r[Qc].values.astype(float)
        if not (np.isfinite(V).all() and np.isfinite(Q).all()): continue
        pf = charge_pinn_feats(V, Q)
        if pf is None: continue
        re = fk.loc[(cell, cc)]
        if isinstance(re, pd.DataFrame): re = re.iloc[0]
        Re_val  = float(re.Re_ohm)*1000
        Rct_val = float(re.Rct_ohm)*1000
        # PINN: 6 features = 4 ICA + Re + Rct
        pf_full = np.concatenate([pf, [Re_val, Rct_val]]).astype(np.float32)
        # 3-channel waveform: V, |I|, T  — Q channel removed (Q_final ≈ SOH×rated → leakage)
        wav = np.stack([resamp(V), resamp(np.abs(I)), resamp(T)]).astype(np.float32)
        rows.append(dict(cell=cell, soh=float(r.soh_pct), wav=wav, pinn=pf_full,
                         Re=Re_val, Rct=Rct_val))
    df = pd.DataFrame(rows)
    print(f"[charge] total={len(df)}  train={(df.cell!=TEST_CELL).sum()}  test={(df.cell==TEST_CELL).sum()}")
    return df

# ── Models ───────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(s, nc=3, drop=0.3, physics=False):
        super().__init__()
        s.feat = nn.Sequential(
            nn.Conv1d(nc,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(drop),
            nn.AdaptiveAvgPool1d(1))
        s.reg = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32,1))
        s.physics = physics
        if physics: s.phy = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,2))
    def forward(s, x):
        f   = s.feat(x).squeeze(-1)
        soh = torch.sigmoid(s.reg(f))
        return (soh, s.phy(f)) if s.physics else (soh, None)

class PINN(nn.Module):
    def __init__(s, nf=4, h=64, L=3):
        super().__init__()
        layers = [nn.Linear(nf,h), nn.Tanh()]
        for _ in range(L-1): layers += [nn.Linear(h,h), nn.Tanh()]
        layers += [nn.Linear(h,1)]
        s.net = nn.Sequential(*layers)
    def forward(s, x): return torch.sigmoid(s.net(x))

def pinn_loss(model, x, y, lm=0.1, lb=0.1):
    p  = model(x)
    xp = x.detach().clone().requires_grad_(True)
    g  = torch.autograd.grad(model(xp).sum(), xp, create_graph=True)[0]
    return ((p-y)**2).mean() + lm*torch.relu(-g).pow(2).mean() + lb*torch.relu(0.6-p).pow(2).mean()

# ── Training functions ────────────────────────────────────────────────────────
def train_cnn(Wtr_n, ytr_t, phytr, Wte_n, physics, drop=0.3, lr=1e-3, wd=1e-3, ep_max=150, seed=42):
    torch.manual_seed(seed)
    net = CNN(nc=3, drop=drop, physics=physics)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n = Wtr_n.shape[0]; idx = np.arange(n)
    for ep in range(ep_max):
        net.train(); np.random.seed(ep); np.random.shuffle(idx)
        for b in range(0, n, 32):
            bi = idx[b:b+32]; soh, phy = net(Wtr_n[bi])
            loss = ((soh - ytr_t[bi])**2).mean()
            if physics and phy is not None:
                loss = loss + 0.01*((phy - phytr[bi])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        yp_tr = net(Wtr_n)[0].numpy().ravel() * 100
        yp_te = net(Wte_n)[0].numpy().ravel() * 100
    return net, yp_tr, yp_te

def train_pinn(Xtr, ytr_raw, Xev, Xte, h=64, L=3, lm=0.1, lb=0.1, ep=3000, seed=42):
    nstd = np.array([0.002, 0.010, 0.002, 0.010, 0.5, 1.0], np.float32)
    rng  = np.random.default_rng(seed)
    Xa=[Xtr]; Ya=[ytr_raw]
    for _ in range(30):
        Xa.append(Xtr + rng.normal(0,1,Xtr.shape).astype(np.float32)*nstd)
        Ya.append(ytr_raw)
    sc    = MinMaxScaler().fit(np.vstack(Xa))
    Xtr_n = torch.tensor(sc.transform(np.vstack(Xa)).astype(np.float32))
    ytr_n = torch.tensor(np.concatenate(Ya)).view(-1,1)
    Xev_n = torch.tensor(sc.transform(Xev).astype(np.float32))
    Xte_n = torch.tensor(sc.transform(Xte).astype(np.float32))
    torch.manual_seed(seed)
    model = PINN(nf=6, h=h, L=L)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch   = ReduceLROnPlateau(opt, patience=200, factor=0.5, min_lr=1e-5)
    best=1e9; best_sd=None
    for e in range(ep):
        model.train(); opt.zero_grad()
        loss = pinn_loss(model, Xtr_n, ytr_n, lm=lm, lb=lb)
        loss.backward(); opt.step(); sch.step(loss.detach())
        if loss.item() < best: best=loss.item(); best_sd=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        yp_tr = model(Xev_n).numpy().ravel() * 100
        yp_te = model(Xte_n).numpy().ravel() * 100
    return model, sc, yp_tr, yp_te

# ── LOCO-CV ───────────────────────────────────────────────────────────────────
def loco_cv_cnn(df_tr, physics):
    tag   = "PI-1D-CNN" if physics else "1D-CNN"
    cells = df_tr.cell.unique()
    best_r2=-np.inf; best_cfg=None
    drops  = [0.0, 0.1] if physics else [0.1, 0.3]
    ep_max = 120        if physics else 150
    for drop in drops:
        for lr in [1e-3, 3e-3]:
            for wd in [1e-4, 1e-3]:
                cv_r2s = []
                for val_cell in cells:
                    tr_c = df_tr[df_tr.cell!=val_cell].reset_index(drop=True)
                    va_c = df_tr[df_tr.cell==val_cell].reset_index(drop=True)
                    Wt   = torch.tensor(np.stack(tr_c.wav.values))
                    Wv   = torch.tensor(np.stack(va_c.wav.values))
                    mu   = Wt.mean(dim=(0,2),keepdim=True)
                    sd   = Wt.std(dim=(0,2),keepdim=True)+1e-6
                    Wt_n, Wv_n = (Wt-mu)/sd, (Wv-mu)/sd
                    yt_t = torch.tensor((tr_c.soh.values/100.).astype(np.float32)).view(-1,1)
                    pt   = torch.tensor(tr_c[["Re","Rct"]].values.astype(np.float32))
                    _, _, yp_val = train_cnn(Wt_n, yt_t, pt, Wv_n, physics,
                                             drop=drop, lr=lr, wd=wd, ep_max=ep_max)
                    cv_r2s.append(r2_score(va_c.soh.values, yp_val))
                avg = np.mean(cv_r2s)
                print(f"  [{tag}] drop={drop} lr={lr} wd={wd}  avg_R2={avg:.3f}")
                if avg > best_r2: best_r2=avg; best_cfg=dict(drop=drop,lr=lr,wd=wd,ep_max=ep_max)
    print(f"  [{tag}] BEST: {best_cfg}  CV_R2={best_r2:.3f}")
    return best_cfg

def loco_cv_pinn(df_tr):
    cells = df_tr.cell.unique()
    best_r2=-np.inf; best_cfg=None
    for h in [64, 128]:
        for L in [2, 3]:
            for lm in [0.01, 0.1, 0.5]:
                for lb in [0.05, 0.1]:
                    cv_r2s = []
                    for val_cell in cells:
                        tr_c = df_tr[df_tr.cell!=val_cell].reset_index(drop=True)
                        va_c = df_tr[df_tr.cell==val_cell].reset_index(drop=True)
                        Xtr  = np.vstack(tr_c.pinn.values).astype(np.float32)
                        ytr  = (tr_c.soh.values/100.).astype(np.float32)
                        Xva  = np.vstack(va_c.pinn.values).astype(np.float32)
                        _, _, _, yp_val = train_pinn(Xtr, ytr, Xtr, Xva, h=h, L=L, lm=lm, lb=lb, ep=2000)
                        cv_r2s.append(r2_score(va_c.soh.values, yp_val))
                    avg = np.mean(cv_r2s)
                    print(f"  [PINN] h={h} L={L} lm={lm} lb={lb}  avg_R2={avg:.3f}")
                    if avg > best_r2: best_r2=avg; best_cfg=dict(h=h,L=L,lm=lm,lb=lb)
    print(f"  [PINN] BEST: {best_cfg}  CV_R2={best_r2:.3f}")
    return best_cfg

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = build_charge_table()
    tr  = df[df.cell!=TEST_CELL].reset_index(drop=True)
    te  = df[df.cell==TEST_CELL].reset_index(drop=True)
    yt  = te.soh.values; results={}; preds={}

    # Normalise CNN waveforms using train-set statistics
    Wtr = torch.tensor(np.stack(tr.wav.values))
    Wte = torch.tensor(np.stack(te.wav.values))
    cmu = Wtr.mean(dim=(0,2), keepdim=True)
    csd = Wtr.std(dim=(0,2),  keepdim=True) + 1e-6
    Wtr_n, Wte_n = (Wtr-cmu)/csd, (Wte-cmu)/csd
    ytr_t  = torch.tensor((tr.soh.values/100.).astype(np.float32)).view(-1,1)
    phytr  = torch.tensor(tr[["Re","Rct"]].values.astype(np.float32))

    # ── CNN: reuse best configs from previous LOCO-CV run ────────────────────
    cnn_cfg = dict(drop=0.3, lr=3e-3, wd=1e-4, ep_max=150)   # CV_R2=0.920
    pi_cfg  = dict(drop=0.1, lr=1e-3, wd=1e-3, ep_max=120)   # CV_R2=0.811
    print(f"[1D-CNN]    reusing best: {cnn_cfg}")
    print(f"[PI-1D-CNN] reusing best: {pi_cfg}")

    # ── PINN LOCO-CV (nf=6: ICA + Re + Rct) ─────────────────────────────────
    print("\n=== PINN LOCO-CV (nf=6) ===")
    pinn_cfg = loco_cv_pinn(tr)

    # ── Final training on all B0005/6/7 ──────────────────────────────────────
    print("\n=== Final training ===")

    # 1D-CNN
    net_cnn, yp_tr_cnn, yp_te_cnn = train_cnn(Wtr_n, ytr_t, phytr, Wte_n, False, **cnn_cfg)
    results["1D-CNN"]    = {"train": metrics(tr.soh.values, yp_tr_cnn), "test": metrics(yt, yp_te_cnn)}
    preds["1D-CNN"]      = yp_te_cnn
    torch.save({"state_dict": net_cnn.state_dict(), "mu": cmu, "sd": csd,
                "cfg": cnn_cfg, "physics": False, "nc": 3},
               os.path.join(MODEL_DIR, "1dcnn_charge.pt"))

    # PI-1D-CNN
    net_pi, yp_tr_pi, yp_te_pi = train_cnn(Wtr_n, ytr_t, phytr, Wte_n, True, **pi_cfg)
    results["PI-1D-CNN"] = {"train": metrics(tr.soh.values, yp_tr_pi), "test": metrics(yt, yp_te_pi)}
    preds["PI-1D-CNN"]   = yp_te_pi
    torch.save({"state_dict": net_pi.state_dict(), "mu": cmu, "sd": csd,
                "cfg": pi_cfg, "physics": True, "nc": 3},
               os.path.join(MODEL_DIR, "pi_1dcnn_charge.pt"))

    # PINN
    Xtr      = np.vstack(tr.pinn.values).astype(np.float32)
    ytr_raw  = (tr.soh.values/100.).astype(np.float32)
    Xte      = np.vstack(te.pinn.values).astype(np.float32)
    pinn_m, sc, yp_tr_pinn, yp_te_pinn = train_pinn(Xtr, ytr_raw, Xtr, Xte, **pinn_cfg, ep=3000)
    results["PINN"]      = {"train": metrics(tr.soh.values, yp_tr_pinn), "test": metrics(yt, yp_te_pinn)}
    preds["PINN"]        = yp_te_pinn
    torch.save({"state_dict": pinn_m.state_dict(), "scaler": sc, "cfg": pinn_cfg},
               os.path.join(MODEL_DIR, "pinn_charge.pt"))

    # ── Print & save results ──────────────────────────────────────────────────
    order = sorted(results, key=lambda k: results[k]["test"]["R2"], reverse=True)
    print("\n=== CHARGE BENCHMARK RESULTS (LOCO-CV clean) ===")
    print(f"{'Model':<15}{'split':>6}{'RMSE%':>8}{'MAE%':>8}{'R2':>9}{'Acc3':>8}")
    tbl = []
    for k in order:
        rt, rte = results[k]["train"], results[k]["test"]
        print(f"{k:<15}{'train':>6}{rt['RMSE']:>8.2f}{rt['MAE']:>8.2f}{rt['R2']:>9.3f}{rt['ACC3']:>8.3f}")
        print(f"{'':<15}{'test':>6}{rte['RMSE']:>8.2f}{rte['MAE']:>8.2f}{rte['R2']:>9.3f}{rte['ACC3']:>8.3f}")
        tbl.append(dict(model=k,
                        train_RMSE=round(rt['RMSE'],4), train_R2=round(rt['R2'],4),
                        test_RMSE=round(rte['RMSE'],4), test_MAE=round(rte['MAE'],4),
                        test_R2=round(rte['R2'],4), test_ACC3=round(rte['ACC3'],4)))
    pd.DataFrame(tbl).to_csv(os.path.join(ROOT, "metrics_charge.csv"), index=False)

    # Parity plot
    plt.figure(figsize=(7,7))
    lim = [yt.min()-2, yt.max()+2]; plt.plot(lim, lim, "k--", lw=1, label="y=x")
    cols = dict(zip(order, plt.cm.Set1(np.linspace(0,1,len(order)))))
    for k in order:
        r = results[k]["test"]
        plt.scatter(yt, preds[k], s=20, alpha=0.7, color=cols[k],
                    label=f"{k}: R²={r['R2']:.3f}, RMSE={r['RMSE']:.2f}%")
    plt.xlabel("True SOH (%)"); plt.ylabel("Predicted SOH (%)")
    plt.title("Charge-curve SOH (LOCO-CV clean)\nB0018 hold-out")
    plt.legend(fontsize=9, loc="upper left"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "parity_charge.png"), dpi=130)
    print(f"\nSaved: metrics_charge.csv, parity_charge.png")
    print(f"Charge winner: {order[0]}")

if __name__ == "__main__": main()
