"""
benchmark_charge.py — 3.1.2.1 Charging-curve (full CC-CV) benchmark of the TOP-3
models found on discharge: 1D-CNN, PINN, PI-1D-CNN.

Why charge is the fair/decisive test: on a discharge cycle capacity ~= SOH, so any
capacity-derived feature (CNN cumQ, PINN cap_ratio) trivially predicts SOH. On the
charge curve the cumulative charge depends on starting SOC and the CV tail, so it does
NOT encode SOH — the model must learn curve shape. This makes 3.1.2.1 the unbiased
decider. Same protocol as 3.1.1.2: train B0005/6/7, test B0018, same labels/metrics,
same test cycles (intersection), leakage-free charge label = SOH of the following
discharge (Semin's matched soh_pct).
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn as nn

warnings.filterwarnings("ignore"); np.random.seed(42); torch.manual_seed(42)
ROOT = os.path.dirname(os.path.abspath(__file__))
SVMDIR = os.path.join(ROOT, "Intel-Cup", "models", "SVM")
TRAIN_CELLS=["B0005","B0006","B0007"]; TEST_CELL="B0018"; N_TS=128; RATED=2.0

def cls3(y): y=np.asarray(y,float); return np.where(y>80,0,np.where(y>=70,1,2))
def metrics(yt,yp):
    yt,yp=np.asarray(yt,float),np.asarray(yp,float)
    return dict(RMSE=float(np.sqrt(mean_squared_error(yt,yp))),MAE=float(mean_absolute_error(yt,yp)),
                R2=float(r2_score(yt,yp)),ACC3=float(accuracy_score(cls3(yt),cls3(yp))))
def resamp(x,n=N_TS): x=np.asarray(x,float); return np.interp(np.linspace(0,1,n),np.linspace(0,1,len(x)),x)

def charge_pinn_feats(V, Q):
    """Charge analog of Donghyun's discharge cap_ratio/dv_norm: charge capacity and
    dV/dQ over two CC voltage windows (leakage-free: depends on SOC/CV, not SOH)."""
    feats=[]
    for lo,hi in [(3.90,4.05),(4.05,4.18)]:
        m=(V>=lo)&(V<=hi)
        if m.sum()<3: return None
        q=Q[m]; v=V[m]; cap=q[-1]-q[0]
        if cap<=1e-6: return None
        dvdq=(v[-1]-v[0])/cap
        feats += [cap/RATED, dvdq*RATED]
    return np.array(feats, np.float32)

def build_charge_table():
    w=pd.read_csv(os.path.join(SVMDIR,"nasa_all_cells_charge_waveform_101.csv"))
    f=pd.read_csv(os.path.join(SVMDIR,"nasa_all_cells_charge_features.csv"))
    fk=f.set_index(["cell_id","charge_cycle"])[["Re_ohm","Rct_ohm"]]
    Vc=[f"V_{i:03d}" for i in range(101)]; Ic=[f"I_{i:03d}" for i in range(101)]
    Tc=[f"T_{i:03d}" for i in range(101)]; Qc=[f"Q_{i:03d}" for i in range(101)]
    rows=[]
    for _,r in w.iterrows():
        cell=r.cell_id; cc=int(r.charge_cycle)
        if (cell,cc) not in fk.index: continue
        V=r[Vc].values.astype(float); I=r[Ic].values.astype(float)
        T=r[Tc].values.astype(float); Q=r[Qc].values.astype(float)
        if not np.isfinite(V).all() or not np.isfinite(Q).all(): continue
        pf=charge_pinn_feats(V,Q)
        if pf is None: continue
        re=fk.loc[(cell,cc)]
        if isinstance(re,pd.DataFrame): re=re.iloc[0]
        wav=np.stack([resamp(V),resamp(np.abs(I)),resamp(T),resamp(Q)]).astype(np.float32)
        rows.append(dict(cell=cell, soh=float(r.soh_pct), wav=wav, pinn=pf,
                         Re=float(re.Re_ohm)*1000, Rct=float(re.Rct_ohm)*1000))
    df=pd.DataFrame(rows)
    print(f"[charge] common cycles: total={len(df)}  train={(df.cell!=TEST_CELL).sum()}  test(B0018)={(df.cell==TEST_CELL).sum()}")
    return df

class CNN(nn.Module):
    def __init__(s, nc=4, physics=False):
        super().__init__()
        s.feat=nn.Sequential(nn.Conv1d(nc,16,7,padding=3),nn.BatchNorm1d(16),nn.ReLU(),
            nn.Conv1d(16,32,5,padding=2),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Conv1d(32,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        s.reg=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)); s.physics=physics
        if physics: s.phy=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(s,x):
        f=s.feat(x).squeeze(-1); soh=torch.sigmoid(s.reg(f))
        return (soh, s.phy(f)) if s.physics else (soh,None)

class PINN(nn.Module):
    def __init__(s,nf=4,h=32):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(nf,h),nn.Tanh(),nn.Linear(h,h),nn.Tanh(),
                            nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1))
    def forward(s,x): return torch.sigmoid(s.net(x))

def main():
    df=build_charge_table()
    tr=df[df.cell!=TEST_CELL].reset_index(drop=True); te=df[df.cell==TEST_CELL].reset_index(drop=True)
    yt=te.soh.values; results={}; preds={}

    # ----- CNN / PI-CNN -----
    Wtr=torch.tensor(np.stack(tr.wav.values)); Wte=torch.tensor(np.stack(te.wav.values))
    cmu=Wtr.mean(dim=(0,2),keepdim=True); csd=Wtr.std(dim=(0,2),keepdim=True)+1e-6
    Wtr_n,Wte_n=(Wtr-cmu)/csd,(Wte-cmu)/csd
    ytr=torch.tensor((tr.soh.values/100.).astype(np.float32)).view(-1,1)
    phytr=torch.tensor(tr[["Re","Rct"]].values.astype(np.float32))
    def train_cnn(physics):
        torch.manual_seed(42); net=CNN(4,physics)
        opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-4)
        n=Wtr_n.shape[0]; idx=np.arange(n)
        for ep in range(150):
            net.train(); np.random.seed(ep); np.random.shuffle(idx)
            for b in range(0,n,32):
                bi=idx[b:b+32]; soh,phy=net(Wtr_n[bi]); loss=((soh-ytr[bi])**2).mean()
                if physics: loss=loss+0.01*((phy-phytr[bi])**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return net(Wtr_n)[0].numpy().ravel()*100, net(Wte_n)[0].numpy().ravel()*100
    ytr_np=tr.soh.values
    a,b=train_cnn(False); results["1D-CNN (Evan)"]={"train":metrics(ytr_np,a),"test":metrics(yt,b)}; preds["1D-CNN (Evan)"]=b
    a,b=train_cnn(True);  results["PI-1D-CNN (Evan)"]={"train":metrics(ytr_np,a),"test":metrics(yt,b)}; preds["PI-1D-CNN (Evan)"]=b

    # ----- PINN -----
    Xtr=torch.tensor(np.vstack(tr.pinn.values)); Xte=torch.tensor(np.vstack(te.pinn.values))
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr_n,Xte_n=(Xtr-mu)/sd,(Xte-mu)/sd
    ytrp=torch.tensor((tr.soh.values/100.).astype(np.float32)).view(-1,1)
    g=torch.Generator().manual_seed(0); aX=[Xtr_n]; aY=[ytrp]
    for _ in range(20): aX.append(Xtr_n+0.02*torch.randn(Xtr_n.shape,generator=g)); aY.append(ytrp)
    AX,AY=torch.cat(aX),torch.cat(aY); pinn=PINN(); opt=torch.optim.Adam(pinn.parameters(),lr=1e-3)
    for ep in range(1500):
        opt.zero_grad(); out=pinn(AX)
        loss=((out-AY)**2).mean()+0.1*((out.clamp(max=0)**2).mean()+((out-1).clamp(min=0)**2).mean())
        loss.backward(); opt.step()
    pinn.eval()
    with torch.no_grad():
        yp=pinn(Xte_n).numpy().ravel()*100; yp_tr=pinn(Xtr_n).numpy().ravel()*100
    results["PINN (Donghyun)"]={"train":metrics(ytr_np,yp_tr),"test":metrics(yt,yp)}; preds["PINN (Donghyun)"]=yp

    order=sorted(results,key=lambda k:results[k]["test"]["R2"],reverse=True)
    print("\n=== 3.1.2.1  CHARGE-CURVE BENCHMARK (top-3, train B0005/6/7, test B0018) ===")
    print(f"{'Model':<18}{'split':>6}{'RMSE%':>8}{'MAE%':>8}{'R2':>9}{'Acc3':>8}")
    tbl=[]
    for k in order:
        rt, rte = results[k]["train"], results[k]["test"]
        print(f"{k:<18}{'train':>6}{rt['RMSE']:>8.2f}{rt['MAE']:>8.2f}{rt['R2']:>9.3f}{rt['ACC3']:>8.3f}")
        print(f"{'':<18}{'test':>6}{rte['RMSE']:>8.2f}{rte['MAE']:>8.2f}{rte['R2']:>9.3f}{rte['ACC3']:>8.3f}")
        tbl.append(dict(model=k, train_RMSE=round(rt['RMSE'],4), train_R2=round(rt['R2'],4), train_ACC3=round(rt['ACC3'],4),
            test_RMSE=round(rte['RMSE'],4), test_MAE=round(rte['MAE'],4), test_R2=round(rte['R2'],4), test_ACC3=round(rte['ACC3'],4)))
    pd.DataFrame(tbl).to_csv(os.path.join(ROOT,"benchmark_charge_metrics.csv"),index=False)
    json.dump({"results":results,"order":order,"n_train":int(len(tr)),"n_test":int(len(te))},
              open(os.path.join(ROOT,"benchmark_charge.json"),"w"),indent=2)

    plt.figure(figsize=(8,8)); lim=[yt.min()-2,yt.max()+2]; plt.plot(lim,lim,"k--",lw=1,label="Ideal (y=x)")
    cols=dict(zip(order,plt.cm.Set1(np.linspace(0,1,len(order)))))
    for k in order:
        r=results[k]["test"]; plt.scatter(yt,preds[k],s=20,alpha=0.6,color=cols[k],
            label=f"{k}: test R²={r['R2']:.3f}, RMSE={r['RMSE']:.2f}%")
    plt.xlabel("True SOH (%)"); plt.ylabel("Predicted SOH (%)")
    plt.title("3.1.2.1 Charging-curve SOH (top-3) on B0018 hold-out\n(leakage-free: charge does not encode SOH)")
    plt.legend(fontsize=9,loc="upper left"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ROOT,"benchmark_charge_parity.png"),dpi=130)
    print("\nCharge winner:", order[0]); print("Saved: benchmark_charge_metrics.csv / .json / _parity.png")

if __name__=="__main__": main()
