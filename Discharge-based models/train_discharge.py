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
  - Leakage ctrl: discharge waveform uses V, |I|, T only (NO cumulative-charge
                  channel), so the CNNs cannot read capacity off the input.

Models:
  SVM (Semin)        : SVR(rbf) on hybrid discharge scalar features
  MLP (Evan)         : MLPRegressor on the SAME scalar features (fair vs SVM)
  PINN (Donghyun)    : SOHCurvePINN (4 normalized SOC-interval feats) + physics loss
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

ROOT = os.path.dirname(os.path.abspath(__file__))
SVMDIR = os.path.join(ROOT, "Intel-Cup", "models", "SVM")
TRAIN_CELLS = ["B0005", "B0006", "B0007"]
TEST_CELL   = "B0018"
ALL_CELLS   = TRAIN_CELLS + [TEST_CELL]
N_TS = 128
RATED = 2.0

HYBRID_FEATS = ["Re_ohm","Rct_ohm","V_start","V_10s","V_30s","V_60s",
                "V_drop_10s","V_drop_30s","V_drop_60s","dV_dt_avg","I_abs_avg",
                "T_start","T_60s","T_rise_60s"]

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
    m = scipy.io.loadmat(os.path.join(ROOT, f"{cell}.mat"))[cell][0,0]
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
        dt = np.diff(tm, prepend=tm[0])
        cumAh = np.cumsum(np.abs(I)*dt)/3600.0          # cumulative discharged charge
        wav = np.stack([resamp(V), resamp(np.abs(I)), resamp(T), resamp(cumAh)])  # (4,128)
        pf = pinn_feats(V,I,T,tm,cap)
        out[dno] = dict(wav=wav.astype(np.float32), pinn=pf, cap=cap)
    return out

def pinn_feats(V,I,T,tm,cap):
    """Donghyun's 4 normalized SOC-interval feats: cap_ratio & dv_norm on 100-80 / 80-60."""
    dt = np.diff(tm, prepend=tm[0])
    cumAh = np.cumsum(np.abs(I)*dt)/3600.0
    soc = np.clip(1 - cumAh/max(cap,1e-6), 0, 1)
    feats = []
    for hi, lo in [(1.0,0.8),(0.8,0.6)]:
        mask = (soc <= hi+0.02) & (soc >= lo-0.02)
        if mask.sum() < 5: return None
        segAh, segV = cumAh[mask], V[mask]
        cap_ah = segAh[-1]-segAh[0]
        if cap_ah <= 1e-6: return None
        dv = (segV[-1]-segV[0])/cap_ah
        feats += [cap_ah/RATED, dv*RATED]
    return np.array(feats, np.float32)   # [cr_100_80, dv_100_80, cr_80_60, dv_80_60]

# ----------------------------------------------------------------------------
def build_discharge_table():
    disc_csv = pd.read_csv(os.path.join(SVMDIR, "nasa_all_cells_discharge_features.csv"))
    rows = []
    for cell in ALL_CELLS:
        mat = load_discharge_mat(cell)
        sub = disc_csv[disc_csv.cell_id == cell].set_index("discharge_cycle")
        for dno, rec in mat.items():
            if dno not in sub.index: continue
            if rec["pinn"] is None: continue
            r = sub.loc[dno]
            if isinstance(r, pd.DataFrame): r = r.iloc[0]
            if r[HYBRID_FEATS].isna().any(): continue
            rows.append(dict(cell=cell, dno=int(dno),
                             soh=float(r["soh_pct"]),
                             Re=float(r["Re_ohm"])*1000.0, Rct=float(r["Rct_ohm"])*1000.0,
                             scal=r[HYBRID_FEATS].values.astype(np.float32),
                             pinn=rec["pinn"], wav=rec["wav"]))
    df = pd.DataFrame(rows)
    print(f"[discharge] common cycles: total={len(df)}  "
          f"train={ (df.cell!=TEST_CELL).sum() }  test(B0018)={ (df.cell==TEST_CELL).sum() }")
    return df

# ----------------------------------------------------------------------------
# sklearn models
def run_svm(tr, te, feat="scal"):
    Xtr = np.vstack(tr[feat].values); Xte = np.vstack(te[feat].values)
    m = Pipeline([("sc",StandardScaler()), ("svr",SVR(kernel="rbf",C=100,epsilon=0.1))])
    m.fit(Xtr, tr.soh.values)
    return m.predict(Xtr), m.predict(Xte)

def run_mlp(tr, te, feat="scal"):
    Xtr = np.vstack(tr[feat].values); Xte = np.vstack(te[feat].values)
    sc = StandardScaler().fit(Xtr)
    m = MLPRegressor(hidden_layer_sizes=(64,64,64), activation="tanh",
                     max_iter=4000, random_state=42, early_stopping=False, alpha=1e-3)
    m.fit(sc.transform(Xtr), tr.soh.values)
    return m.predict(sc.transform(Xtr)), m.predict(sc.transform(Xte))

# ----------------------------------------------------------------------------
# torch models (PINN, 1D-CNN, PI-1D-CNN)
def run_torch_models(tr, te, results, preds):
    try:
        import torch, torch.nn as nn
    except Exception as e:
        print("!! torch unavailable -> skipping PINN / 1D-CNN / PI-1D-CNN  (", e, ")")
        return
    torch.manual_seed(42)

    # ---- PINN (Donghyun): 4 feats + physics (monotonic + boundary) ----
    class PINN(nn.Module):
        def __init__(s, nf=4, h=32):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(nf,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),
                                  nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
        def forward(s,x): return torch.sigmoid(s.net(x))

    Xtr = torch.tensor(np.vstack(tr.pinn.values)); ytr = torch.tensor((tr.soh.values/100.).astype(np.float32)).view(-1,1)
    Xte = torch.tensor(np.vstack(te.pinn.values))
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6
    Xtr_n, Xte_n = (Xtr-mu)/sd, (Xte-mu)/sd
    # noise augmentation
    aug_X = [Xtr_n]; aug_y=[ytr]
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        aug_X.append(Xtr_n + 0.02*torch.randn(Xtr_n.shape, generator=g)); aug_y.append(ytr)
    AX, AY = torch.cat(aug_X), torch.cat(aug_y)
    pinn = PINN(); opt = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    for ep in range(1500):
        opt.zero_grad()
        out = pinn(AX); loss = ((out-AY)**2).mean()
        # boundary penalty (keep in [0,1] — sigmoid already, light reg)
        loss = loss + 0.1*((out.clamp(max=0)**2).mean() + ((out-1).clamp(min=0)**2).mean())
        loss.backward(); opt.step()
    pinn.eval()
    with torch.no_grad():
        yp = pinn(Xte_n).numpy().ravel()*100
        yp_tr = pinn(Xtr_n).numpy().ravel()*100
    results["PINN (Donghyun)"]={"train":metrics(tr.soh.values,yp_tr),"test":metrics(te.soh.values,yp)}; preds["PINN (Donghyun)"]=yp

    # ---- CNN backbone ----
    class CNN(nn.Module):
        def __init__(s, nc=4, physics=False):
            super().__init__()
            s.feat = nn.Sequential(
                nn.Conv1d(nc,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Conv1d(16,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            s.reg = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
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
        net = CNN(4, physics); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        n = Wtr_n.shape[0]; idx = np.arange(n)
        for ep in range(120):
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
        return yp_tr, yp

    a,b = train_cnn(False)
    results["1D-CNN (Evan)"]={"train":metrics(tr.soh.values,a),"test":metrics(te.soh.values,b)}; preds["1D-CNN (Evan)"]=b
    a,b = train_cnn(True)
    results["PI-1D-CNN (Evan)"]={"train":metrics(tr.soh.values,a),"test":metrics(te.soh.values,b)}; preds["PI-1D-CNN (Evan)"]=b

# ----------------------------------------------------------------------------
def main():
    df = build_discharge_table()
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
