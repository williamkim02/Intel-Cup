# -*- coding: utf-8 -*-
"""Parity: plain 1D-CNN vs PI-1D-CNN vs Segment PINN on held-out B0018."""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO=r"D:\Personal Projects\Intel-Cup-repo"; MD=os.path.join(REPO,"Discharge-based models","models")
N_TS,WIN,STRIDE=128,13,6

class PINN(nn.Module):
    def __init__(s,nf=4,h=128,L=3):
        super().__init__(); Ls=[nn.Linear(nf,h),nn.Tanh()]
        for _ in range(L-1): Ls+=[nn.Linear(h,h),nn.Tanh()]
        Ls+=[nn.Linear(h,1)]; s.net=nn.Sequential(*Ls)
    def forward(s,x): return torch.sigmoid(s.net(x))
def feat(nc,drop):
    return nn.Sequential(nn.Conv1d(nc,16,7,padding=3),nn.BatchNorm1d(16),nn.ReLU(),
                         nn.Conv1d(16,32,5,padding=2),nn.BatchNorm1d(32),nn.ReLU(),
                         nn.Conv1d(32,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),
                         nn.Dropout(drop),nn.AdaptiveAvgPool1d(1))
class CNNplain(nn.Module):
    def __init__(s,drop=0.1): super().__init__(); s.feat=feat(3,drop); s.reg=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(drop),nn.Linear(32,1))
    def forward(s,x): return torch.sigmoid(s.reg(s.feat(x).squeeze(-1)))
class CNNpi(nn.Module):
    def __init__(s,drop=0.1):
        super().__init__(); s.feat=feat(3,drop)
        s.reg=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(drop),nn.Linear(32,1))
        s.phy=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(s,x): f=s.feat(x).squeeze(-1); return torch.sigmoid(s.reg(f)),s.phy(f)

z=np.load(os.path.join(REPO,"Discharge-based models","data","discharge_dataset.npz"),allow_pickle=True)
wav=z['wav']; soh=z['soh']; cell=z['cell']; te=np.where(cell=='B0018')[0]
P=[];Wv=[];cy=[];tr=[]
for i in te:
    V=wav[i,0]
    for ws in range(0,N_TS-WIN+1,STRIDE):
        we=ws+WIN; P.append([float(V[ws]),float(V[we-1]),float(V[ws]-V[we-1]),1.0-(ws+we)/2/N_TS])
        seg=wav[i,:,ws:we]; Wv.append(np.stack([np.interp(np.linspace(0,1,N_TS),np.linspace(0,1,WIN),seg[c]) for c in range(3)]))
        cy.append(i); tr.append(soh[i])
P=np.array(P,np.float32); Wv=torch.tensor(np.array(Wv,np.float32)); cy=np.array(cy); tr=np.array(tr)

# PINN
dp=torch.load(os.path.join(MD,"pinn_segment.pt"),weights_only=False,map_location="cpu")
pm=PINN(4,dp['cfg']['h'],dp['cfg']['L']); pm.load_state_dict(dp['state_dict']); pm.eval()
with torch.no_grad(): pinn=pm(torch.tensor(dp['scaler'].transform(P).astype(np.float32))).numpy().ravel()*100
# PI-1D-CNN
dpi=torch.load(os.path.join(MD,"pi_1dcnn_segment.pt"),weights_only=False,map_location="cpu")
pic=CNNpi(dpi['cfg'].get('drop',0.1)); pic.load_state_dict(dpi['state_dict']); pic.eval()
with torch.no_grad(): picnn=pic((Wv-dpi['mu'])/dpi['sd'])[0].numpy().ravel()*100
# plain CNN
dc=torch.load(os.path.join(MD,"cnn_segment.pt"),weights_only=False,map_location="cpu")
pc=CNNplain(dc['cfg'].get('drop',0.1)); pc.load_state_dict(dc['state_dict']); pc.eval()
with torch.no_grad(): pcnn=pc((Wv-dc['mu'])/dc['sd']).numpy().ravel()*100

df=pd.DataFrame({'cyc':cy,'t':tr,'pinn':pinn,'pi':picnn,'cnn':pcnn})
g=df.groupby('cyc').agg(t=('t','first'),pinn=('pinn','mean'),pi=('pi','mean'),cnn=('cnn','mean')).reset_index()
r_pinn=r2_score(g.t,g.pinn); r_pi=r2_score(g.t,g.pi); r_cnn=r2_score(g.t,g.cnn)
print(f"PINN={r_pinn:.3f}  PI-1D-CNN={r_pi:.3f}  plain 1D-CNN={r_cnn:.3f}")

fig,ax=plt.subplots(figsize=(7.6,6.4)); lim=[g.t.min()-3,g.t.max()+3]
ax.plot(lim,lim,'--',color='#9aa4b2',lw=1.3,label='ideal (y = x)')
ax.scatter(g.t,g.cnn,s=40,color='#c9a27e',edgecolor='white',lw=.5,label=f'1D-CNN — waveform, no physics   (R² = {r_cnn:.2f})')
ax.scatter(g.t,g.pi,s=40,color='#e08a3c',edgecolor='white',lw=.5,label=f'PI-1D-CNN — waveform + physics   (R² = {r_pi:.2f})')
ax.scatter(g.t,g.pinn,s=40,color='#1b2a5b',edgecolor='white',lw=.5,label=f'Segment PINN — features + physics   (R² = {r_pinn:.2f})')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
ax.set_xlabel('True SOH on B0018 (%)',fontsize=12); ax.set_ylabel('Predicted SOH (%)',fontsize=12)
ax.set_title('Held-out cell B0018: waveform CNNs vs the segment PINN',fontsize=12,fontweight='bold',pad=12)
ax.legend(fontsize=9.5,loc='upper left',framealpha=.95); ax.grid(alpha=.25)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"three_model_parity.png"),dpi=150)
print("saved three_model_parity.png")
