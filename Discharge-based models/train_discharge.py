"""
benchmark_models.py — Fair, unbiased model comparison for SOH estimation.

Report sections 3.1.1.2 (discharge) and 3.1.2.1 (charge, top-3).

FAIRNESS PROTOCOL (identical for every model):
  - Train cells : B0005, B0006, B0007   | Test cell: B0018 (never seen)
  - Label       : Semin's soh_pct (capacity_Ah / 2.0 * 100) — ONE shared source
  - Test set    : the SAME B0018 cycles for every model (intersection of cycles
                  that every representation can produce) — so all models are
                  scored on identical points.
  - Metrics     : RMSE(%), MAE(%), R2, 3-class accuracy
                  (Good > 80, Marginal 70-80, Replace < 70)
  - Leakage ctrl:
      * Discharge waveform uses V, |I|, T only (3 channels, NO cumulative-charge).
      * PINN uses time-fixed voltage features (V_10s, V_30s, V_60s, V_120s)
        instead of SOC-window capacity ratios, which directly encode SOH.

Models:
  SVM (Semin)        : SVR(rbf) on hybrid discharge scalar features
  MLP (Evan)         : MLPRegressor on the SAME scalar features (fair vs SVM)
  PINN (Donghyun)    : SOHCurvePINN (4 time-fixed voltage feats) + physics loss
  1D-CNN (Evan)      : Conv1d backbone on (3,128) V/I/T waveform
  PI-1D-CNN (Evan)   : same backbone + physics head (Re/Rct from NASA impedance)
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import scipy.io
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = os.path.dirname(os.path.abspath(__file__))
MATDIR  = os.path.join(ROOT, "..", "data", "NASA_raw")
SVMDIR  = os.path.join(ROOT, "data")
TRAIN_CELLS = ["B0005", "B0006", "B0007"]
TEST_CELL   = "B0018"
ALL_CELLS   = TRAIN_CELLS + [TEST_CELL]
N_TS = 128
RATED = 2.0

# Discharge-CURVE features only (NO impedance Re/Rct) — agreed: compare on the
# discharge curve, not on separately-measured resistance.
CURVE_FEATS = ["V_start","V_10s","V_30s","V_60s",
                "V_drop_10s","V_drop_30s","V_drop_60s","dV_dt_avg","I_abs_avg",
                "T_start","T_60s","T_rise_60s"]

# PINN features: time-fixed voltage points (leakage-free).
# V at t=10/30/60/120s does NOT encode total capacity — it reflects IR drop and
# polarisation kinetics at a fixed elapsed time. All features increase monotonically
# with SOH, so the monotonicity physics loss is correctly oriented.
PINN_FEATS = ["V_10s", "V_30s", "V_60s", "V_120s"]

# ----------------------------------------------------------------------------
def cls3(y):
    y = np.asarray(y, float)
    return np.where(y > 80, 0, np.where(y >= 70, 1, 2))   # Good/Marginal/Replace

def metrics(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    return dict(RMSE=float(np.sqrt(mean_squared_error(yt,yp))),
                MAE=float(mean_absolute_error(yt,yp)),
                R2=float(r2_score(yt,yp)),
                ACC3=float(accuracy_score(cls3(yt), cls3(yp))))

def resamp(x, n=N_TS):
    x = np.asarray(x, float)
    return np.interp(np.linspace(0,1,n), np.linspace(0,1,len(x)), x)

# ----------------------------------------------------------------------------
def load_discharge_mat(cell):
    """Per discharge cycle: waveform (V,|I|,T), pinn feats, cap_total. Keyed by dno."""
    m = scipy.io.loadmat(os.path.join(MATDIR, f"{cell}.mat"))[cell][0,0]
    cyc = m["cycle"][0]; out = {}; dno = 0
    for i in range(cyc.shape[0]):
        c = cyc[i]
        if "discharge" not in str(c["type"][0]).lower():
            continue
        dno += 1
        d = c["data"][0,0]
        V = d["Voltage_measured"].flatten().astype(float)
        I = d["Current_measured"].flatten().astype(float)
        T = d["Temperature_measured"].flatten().astype(float)
        tm= d["Time"].flatten().astype(float)
        cap = float(d["Capacity"].flatten()[0])
        if len(V) < 10 or not (np.isfinite(V).all() and np.isfinite(I).all()):
            continue
        wav = np.stack([resamp(V), resamp(np.abs(I)), resamp(T)])  # (3,128) NO cumAh
        out[dno] = dict(wav=wav.astype(np.float32), cap=cap)
    return out

# ----------------------------------------------------------------------------
def build_discharge_table():
    disc_csv = pd.read_csv(os.path.join(SVMDIR, "nasa_all_cells_discharge_features.csv"))
    rows = []
    for cell in ALL_CELLS:
        mat = load_discharge_mat(cell)
        sub = disc_csv[disc_csv.cell_id == cell].set_index("discharge_cycle")
        for dno, rec in mat.items():
            if dno not in sub.index: continue
            r = sub.loc[dno]
            if isinstance(r, pd.DataFrame): r = r.iloc[0]
            if r[CURVE_FEATS].isna().any(): continue
            if r[PINN_FEATS].isna().any(): continue
            rows.append(dict(cell=cell, dno=int(dno),
                             soh=float(r["soh_pct"]),
                             Re=float(r["Re_ohm"])*1000.0, Rct=float(r["Rct_ohm"])*1000.0,
                             scal=r[CURVE_FEATS].values.astype(np.float32),
                             pinn=r[PINN_FEATS].values.astype(np.float32),
                             wav=rec["wav"]))
    df = pd.DataFrame(rows)
    print(f"[discharge] common cycles: total={len(df)}  "
          f"train={ (df.cell!=TEST_CELL).sum() }  test(B0018)={ (df.cell==TEST_CELL).sum() }")
    return df

# ----------------------------------------------------------------------------
# sklearn models
def run_svm(tr, te, feat="scal"):
    import joblib
    Xtr = np.vstack(tr[feat].values); Xte = np.vstack(te[feat].values)
    # C=400, eps=0.05: selected by LOCO-CV on B0005/B0006/B0007 (CV_R²=0.482)
    m = Pipeline([("sc",StandardScaler()), ("svr",SVR(kernel="rbf",C=400,epsilon=0.05))])
    m.fit(Xtr, tr.soh.values)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump({"model": m, "features": CURVE_FEATS},
                os.path.join(ROOT, "models/svm_discharge.joblib"))
    return m.predict(Xtr), m.predict(Xte)

def run_mlp(tr, te, feat="scal"):
    Xtr = np.vstack(tr[feat].values); Xte = np.vstack(te[feat].values)
    import joblib
    # (128,64), alpha=5e-3: selected by LOCO-CV on B0005/B0006/B0007 (CV_R²=0.438)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("mlp", MLPRegressor(hidden_layer_sizes=(128,64), activation="relu",
                                          solver="lbfgs", max_iter=3000, random_state=42, alpha=5e-3))])
    pipe.fit(Xtr, tr.soh.values)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump({"model": pipe, "features": CURVE_FEATS},
                os.path.join(ROOT, "models/mlp_discharge.joblib"))
    return pipe.predict(Xtr), pipe.predict(Xte)

# ----------------------------------------------------------------------------
# torch models (PINN, 1D-CNN, PI-1D-CNN)
def run_torch_models(tr, te, results, preds):
    try:
        import torch, torch.nn as nn
    except Exception as e:
        print("!! torch unavailable -> skipping PINN / 1D-CNN / PI-1D-CNN  (", e, ")")
        return
    torch.manual_seed(42)

    # ---- PINN (Donghyun): FAITHFUL recipe from train_soh_universal.py ----
    #   MinMaxScaler | L = MSE + 0.1*L_mono + 0.1*L_bound | x30 noise | 3000 epochs
    #   h=128, L=3: selected by LOCO-CV (CV_R²=0.652). lm=0.1 beats lm=0.5 in LOCO-CV.
    from sklearn.preprocessing import MinMaxScaler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import copy
    class SOHCurvePINN(nn.Module):
        def __init__(s, nf=4, h=128, L=3):
            super().__init__()
            layers=[nn.Linear(nf,h),nn.Tanh()]
            for _ in range(L-1): layers += [nn.Linear(h,h),nn.Tanh()]
            layers.append(nn.Linear(h,1)); s.net=nn.Sequential(*layers)
        def forward(s,x): return torch.sigmoid(s.net(x))
    def pinn_loss(model, x, y, lm=0.5, lb=0.1):
        pred=model(x); l_data=((pred-y)**2).mean()
        xp=x.detach().clone().requires_grad_(True)
        grads=torch.autograd.grad(model(xp).sum(), xp, create_graph=True)[0]
        l_mono=torch.relu(-grads).pow(2).mean()
        l_bound=torch.relu(0.6-pred).pow(2).mean()
        return l_data + lm*l_mono + lb*l_bound
    # PINN features: V_10s, V_30s, V_60s, V_120s (leakage-free, all monotone with SOH)
    Xraw=np.vstack(tr.pinn.values).astype(np.float32); yraw=(tr.soh.values/100.).astype(np.float32)
    nstd=np.array([0.002,0.002,0.002,0.002],np.float32)           # ~2mV voltage noise
    rng=np.random.default_rng(42); Xa=[Xraw]; Ya=[yraw]
    for _ in range(30):
        Xa.append(Xraw+rng.normal(0,1,Xraw.shape).astype(np.float32)*nstd); Ya.append(yraw)
    Xaug=np.vstack(Xa); yaug=np.concatenate(Ya)
    sc=MinMaxScaler().fit(Xaug)
    Xtr_n=torch.tensor(sc.transform(Xaug).astype(np.float32)); ytr_n=torch.tensor(yaug).view(-1,1)
    Xtr_eval=torch.tensor(sc.transform(Xraw).astype(np.float32))
    Xte_n=torch.tensor(sc.transform(np.vstack(te.pinn.values)).astype(np.float32))
    torch.manual_seed(42); pinn=SOHCurvePINN(4)
    opt=torch.optim.Adam(pinn.parameters(),lr=1e-3); sched=ReduceLROnPlateau(opt,patience=200,factor=0.5,min_lr=1e-5)
    best=1e9; best_state=None
    for ep in range(int(os.environ.get("PINN_EPOCHS","3000"))):
        pinn.train(); opt.zero_grad(); loss=pinn_loss(pinn,Xtr_n,ytr_n,lm=0.1,lb=0.1)
        loss.backward(); opt.step(); sched.step(loss.detach())
        if loss.item()<best: best=loss.item(); best_state=copy.deepcopy(pinn.state_dict())
    pinn.load_state_dict(best_state); pinn.eval()
    with torch.no_grad():
        yp=pinn(Xte_n).numpy().ravel()*100; yp_tr=pinn(Xtr_eval).numpy().ravel()*100
    results["PINN (Donghyun)"]={"train":metrics(tr.soh.values,yp_tr),"test":metrics(te.soh.values,yp)}; preds["PINN (Donghyun)"]=yp
    # Save PINN model
    os.makedirs(os.path.join(ROOT,"models"), exist_ok=True)
    torch.save({"state_dict": best_state,
                "scaler_min": sc.data_min_.astype(np.float32),
                "scaler_range": sc.data_range_.astype(np.float32),
                "pinn_feats": PINN_FEATS},
               os.path.join(ROOT,"models/pinn_discharge.pt"))

    # ---- CNN backbone ----
    class CNN(nn.Module):
        def __init__(s, nc=3, physics=False):
            super().__init__()
            # Non-physics: drop=0.3/fc=0.6 by LOCO-CV (CV_R²=0.755).
            # Physics CNN: drop=0.0, Re/Rct auxiliary loss provides regularisation.
            drop = 0.0 if physics else 0.3
            s.feat = nn.Sequential(
                nn.Conv1d(nc,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(drop),
                nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(drop),
                nn.Conv1d(32,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            fc_drop = 0.0 if physics else 0.6
            s.reg = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(fc_drop), nn.Linear(32,1))
            s.physics = physics
            if physics: s.phy = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
        def forward(s,x):
            f = s.feat(x).squeeze(-1)
            soh = torch.sigmoid(s.reg(f))
            return (soh, s.phy(f)) if s.physics else (soh, None)

    Wtr = torch.tensor(np.stack(tr.wav.values)); Wte = torch.tensor(np.stack(te.wav.values))
    # per-channel z-norm from train
    cmu = Wtr.mean(dim=(0,2),keepdim=True); csd = Wtr.std(dim=(0,2),keepdim=True)+1e-6
    Wtr_n, Wte_n = (Wtr-cmu)/csd, (Wte-cmu)/csd
    ytr_w = torch.tensor((tr.soh.values/100.).astype(np.float32)).view(-1,1)
    phytr = torch.tensor(tr[["Re","Rct"]].values.astype(np.float32))

    def train_cnn(physics):
        torch.manual_seed(42)
        # lr/wd/ep by LOCO-CV: physics lr=3e-3,wd=1e-4,ep=120 (CV_R²=0.750)
        #                      non-physics lr=1e-3,wd=1e-3,ep=200 (CV_R²=0.755)
        wd = 1e-4 if physics else 1e-3
        ep_max = 120 if physics else 200
        lr = 3e-3 if physics else 1e-3
        net = CNN(3, physics); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        n = Wtr_n.shape[0]; idx = np.arange(n)
        for ep in range(ep_max):
            net.train(); np.random.seed(ep); np.random.shuffle(idx)
            for b in range(0, n, 32):
                bi = idx[b:b+32]
                xb = Wtr_n[bi]; yb = ytr_w[bi]
                soh, phy = net(xb)
                loss = ((soh-yb)**2).mean()
                if physics:
                    loss = loss + 0.01*((phy-phytr[bi])**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            yp = net(Wte_n)[0].numpy().ravel()*100
            yp_tr = net(Wtr_n)[0].numpy().ravel()*100
        # Save model
        fname = "pi_1dcnn_discharge.pt" if physics else "1dcnn_discharge.pt"
        torch.save({"state_dict": net.state_dict(),
                    "chan_mu": cmu.squeeze().numpy(),
                    "chan_sd": csd.squeeze().numpy()},
                   os.path.join(ROOT, "models", fname))
        return yp_tr, yp

    a,b = train_cnn(False)
    results["1D-CNN (Evan)"]={"train":metrics(tr.soh.values,a),"test":metrics(te.soh.values,b)}; preds["1D-CNN (Evan)"]=b
    a,b = train_cnn(True)
    results["PI-1D-CNN (Evan)"]={"train":metrics(tr.soh.values,a),"test":metrics(te.soh.values,b)}; preds["PI-1D-CNN (Evan)"]=b

# ----------------------------------------------------------------------------
def main():
    df = build_discharge_table()

    # Save updated npz so plot_all_models.py uses the new leakage-free features
    is_test = (df.cell == TEST_CELL).values
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    np.savez(os.path.join(ROOT, "data/discharge_dataset.npz"),
             soh=df.soh.values,
             wav=np.stack(df.wav.values),
             pinn=np.stack(df.pinn.values),
             scal=np.stack(df.scal.values),
             re=df.Re.values.astype(np.float32),
             rct=df.Rct.values.astype(np.float32),
             is_test=is_test,
             cell=np.array(df.cell.values))
    print("Saved: data/discharge_dataset.npz")

    tr = df[df.cell != TEST_CELL].reset_index(drop=True)
    te = df[df.cell == TEST_CELL].reset_index(drop=True)
    yt = te.soh.values

    results, preds = {}, {}
    ytr = tr.soh.values
    print("Running SVM (Semin)...");  a,b = run_svm(tr,te); results["SVM (Semin)"]={"train":metrics(ytr,a),"test":metrics(yt,b)}; preds["SVM (Semin)"]=b
    print("Running MLP (Evan)...");   a,b = run_mlp(tr,te); results["MLP (Evan)"]={"train":metrics(ytr,a),"test":metrics(yt,b)}; preds["MLP (Evan)"]=b
    print("Running torch models...")
    run_torch_models(tr, te, results, preds)

    # ---- table (train + test) ----
    order = sorted(results, key=lambda k: results[k]["test"]["R2"], reverse=True)
    print("\n=== 3.1.1.2  DISCHARGE-CURVE BENCHMARK (train B0005/6/7, test B0018) ===")
    print(f"{'Model':<18}{'split':>6}{'RMSE%':>8}{'MAE%':>8}{'R2':>9}{'Acc3':>8}")
    tbl=[]
    for k in order:
        rt, rte = results[k]["train"], results[k]["test"]
        print(f"{k:<18}{'train':>6}{rt['RMSE']:>8.2f}{rt['MAE']:>8.2f}{rt['R2']:>9.3f}{rt['ACC3']:>8.3f}")
        print(f"{'':<18}{'test':>6}{rte['RMSE']:>8.2f}{rte['MAE']:>8.2f}{rte['R2']:>9.3f}{rte['ACC3']:>8.3f}")
        tbl.append(dict(model=k,
            train_RMSE=round(rt['RMSE'],4), train_R2=round(rt['R2'],4), train_ACC3=round(rt['ACC3'],4),
            test_RMSE=round(rte['RMSE'],4), test_MAE=round(rte['MAE'],4),
            test_R2=round(rte['R2'],4), test_ACC3=round(rte['ACC3'],4)))
    pd.DataFrame(tbl).to_csv(os.path.join(ROOT,"benchmark_discharge_metrics.csv"), index=False)
    json.dump({"results":results,"order":order,
               "n_train":int(len(tr)),"n_test":int(len(te))},
              open(os.path.join(ROOT,"benchmark_discharge.json"),"w"), indent=2)

    # ---- combined parity plot ----
    plt.figure(figsize=(8,8))
    lim=[yt.min()-2, yt.max()+2]
    plt.plot(lim,lim,"k--",lw=1,label="Ideal (y=x)")
    colors=dict(zip(order, plt.cm.tab10(np.linspace(0,1,len(order)))))
    for k in order:
        r=results[k]["test"]
        plt.scatter(yt, preds[k], s=18, alpha=0.6, color=colors[k],
                    label=f"{k}: test R²={r['R2']:.3f}, RMSE={r['RMSE']:.2f}%")
    plt.xlabel("True SOH (%)"); plt.ylabel("Predicted SOH (%)")
    plt.title("3.1.1.2 Discharge-curve SOH: all models on B0018 hold-out\n"
              "(same train/test split, same labels & test cycles; each model's native input)")
    plt.legend(fontsize=8, loc="upper left"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ROOT,"benchmark_discharge_parity.png"), dpi=130)
    print("\nTop-3 by R2:", order[:3])
    print("Saved: benchmark_discharge_metrics.csv / .json / _parity.png")

if __name__ == "__main__":
    main()
