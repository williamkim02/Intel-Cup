"""
train_soh_universal.py
배터리 크기에 무관한 Universal SOH PINN

핵심 아이디어:
  기존 모델은 절대값 Ah로 학습 → 2Ah 셀만 유효
  이 모델은 정격용량으로 정규화 -> 어떤 크기 배터리도 동일 모델 사용 가능

정규화 방법:
  cap_ratio  = cap_ah / rated_cap    (무차원, 새 배터리 ~0.20)
  dv_norm    = dv_dah * rated_cap    (V 단위, 용량 스케일 제거)

추론 시 사용법:
  rated_cap = 사용자 입력 (배터리 라벨의 Ah 값)
  → 측정한 cap_ah / rated_cap, dv_dah * rated_cap
  → 동일 모델로 SOH 예측

학습: B0005/B0006/B0007 (NASA rated_cap = 2.0 Ah)
테스트: B0018
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from pinn_model import SOHCurvePINN

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAT_DIR   = os.path.join(os.path.dirname(__file__), "data", "NASA_raw")
CSV_DIR   = os.path.join(os.path.dirname(__file__), "data", "NASA_raw")

TRAIN_CELLS  = ["B0005", "B0006", "B0007"]
TEST_CELL    = "B0018"
NASA_RATED   = 2.0      # NASA 18650 셀 정격용량 (Ah)

SOC_BREAKS = [1.00, 0.80, 0.60]
SOC_TOL    = 0.02

INPUT_FEATURES = [
    "cap_ratio_100_80",   # cap_ah / rated_cap  (무차원)
    "cap_ratio_80_60",
    "dv_norm_100_80",     # dv_dah * rated_cap  (V)
    "dv_norm_80_60",
]

N_NOISE   = 30
NOISE_STD = {"cap_ratio": 0.0005, "dv_norm": 0.020}
LAMBDA_MONO  = 0.5
LAMBDA_BOUND = 0.1
EPOCHS   = 3000
LR       = 1e-3
SAVE_PATH = "soh_universal_model.pth"


# ── .mat 파싱 ──────────────────────────────────────────────────────────────────

def extract_features_normalized(mat_path, cell_name, gt_csv_path, rated_cap):
    """
    .mat 방전 곡선 → 정규화된 feature 추출.
    cap_ratio = cap_ah / rated_cap
    dv_norm   = dv_dah * rated_cap
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=False)
    top = mat[cell_name][0, 0]

    df_gt  = pd.read_csv(gt_csv_path)
    gt_map = df_gt.groupby("discharge_cycle")["soh_pct"].mean().to_dict()

    rows    = []
    disc_no = 0

    for i in range(top["cycle"].shape[1]):
        c     = top["cycle"][0, i]
        ctype = str(c["type"][0]).strip().lower()
        if "discharge" not in ctype:
            continue
        disc_no += 1

        soh_pct = gt_map.get(disc_no, np.nan)
        if np.isnan(soh_pct):
            continue

        data = c["data"][0, 0]
        V    = data["Voltage_measured"].flatten().astype(np.float64)
        I    = data["Current_measured"].flatten().astype(np.float64)
        T    = data["Time"].flatten().astype(np.float64)
        cap_total = float(data["Capacity"].flatten()[0])

        feats = _extract_intervals(V, I, T, cap_total, rated_cap)
        if feats is None:
            continue

        rows.append({
            "cell":     cell_name,
            "disc_idx": disc_no,
            "soh":      soh_pct / 100.0,
            **feats,
        })

    return pd.DataFrame(rows)


def _extract_intervals(V, I, T, cap_total, rated_cap):
    dt    = np.diff(T, prepend=T[0])
    cumAh = np.cumsum(np.abs(I) * dt) / 3600.0
    soc   = np.clip(1.0 - cumAh / max(cap_total, 1e-6), 0.0, 1.0)

    feats = {}
    intervals = list(zip(SOC_BREAKS[:-1], SOC_BREAKS[1:]))

    for soc_hi, soc_lo in intervals:
        label = f"{int(soc_hi*100)}_{int(soc_lo*100)}"
        mask  = (soc <= soc_hi + SOC_TOL) & (soc >= soc_lo - SOC_TOL)
        if mask.sum() < 5:
            return None

        seg_Ah = cumAh[mask]
        seg_V  = V[mask]
        cap_ah = seg_Ah[-1] - seg_Ah[0]
        dv_dah = (seg_V[-1] - seg_V[0]) / cap_ah if cap_ah > 1e-6 else np.nan

        if np.isnan(dv_dah):
            return None

        # 정규화
        feats[f"cap_ratio_{label}"] = cap_ah / rated_cap
        feats[f"dv_norm_{label}"]   = dv_dah * rated_cap

    return feats


# ── 데이터 준비 ───────────────────────────────────────────────────────────────

def prepare_data():
    rng = np.random.default_rng(42)

    print("Feature 추출 (정규화)...")
    train_frames = []
    for cell in TRAIN_CELLS:
        df = extract_features_normalized(
            os.path.join(MAT_DIR, f"{cell}.mat"), cell,
            os.path.join(CSV_DIR,  f"plecs_params_{cell}.csv"), NASA_RATED,
        )
        print(f"  {cell}: {len(df)} 사이클  "
              f"SOH {df['soh'].min()*100:.1f}~{df['soh'].max()*100:.1f}%")
        train_frames.append(df)

    test_df = extract_features_normalized(
        os.path.join(MAT_DIR, f"{TEST_CELL}.mat"), TEST_CELL,
        os.path.join(CSV_DIR,  f"plecs_params_{TEST_CELL}.csv"), NASA_RATED,
    )
    print(f"  {TEST_CELL}: {len(test_df)} 사이클  "
          f"SOH {test_df['soh'].min()*100:.1f}~{test_df['soh'].max()*100:.1f}%")

    train_df = pd.concat(train_frames, ignore_index=True)

    # 정규화된 feature 범위 출력
    print("\n[정규화 feature 범위]")
    for f in INPUT_FEATURES:
        print(f"  {f}: {train_df[f].min():.4f} ~ {train_df[f].max():.4f}")

    print(f"\n  cap_ratio ~= 0.20 이면 건강한 20% SOC 구간 (SOH=100% 기준)")
    print(f"  -> 어떤 크기 배터리든 동일한 범위로 수렴\n")

    X_raw = train_df[INPUT_FEATURES].values.astype(np.float32)
    y_raw = train_df["soh"].values.astype(np.float32)

    # 노이즈 증강
    X_list, y_list = [X_raw], [y_raw]
    for _ in range(N_NOISE):
        noisy = X_raw.copy()
        for j, f in enumerate(INPUT_FEATURES):
            key = "cap_ratio" if "cap_ratio" in f else "dv_norm"
            noisy[:, j] += rng.normal(0, NOISE_STD[key], size=len(X_raw))
        X_list.append(noisy)
        y_list.append(y_raw)

    X_train = np.vstack(X_list).astype(np.float32)
    y_train = np.concatenate(y_list).astype(np.float32)
    print(f"학습 샘플: {len(X_train):,}  ({len(X_raw)} 원본 x {N_NOISE+1} 버전)")

    scaler    = MinMaxScaler()
    X_train_n = scaler.fit_transform(X_train).astype(np.float32)
    X_test_n  = scaler.transform(
        test_df[INPUT_FEATURES].values.astype(np.float32)
    ).astype(np.float32)
    y_test = test_df["soh"].values.astype(np.float32)

    return X_train_n, y_train, X_test_n, y_test, scaler, test_df


# ── 손실 함수 ─────────────────────────────────────────────────────────────────

def compute_loss(model, x_data, y_data, lm=LAMBDA_MONO, lb=LAMBDA_BOUND):
    soh_pred = model(x_data)
    l_data   = nn.functional.mse_loss(soh_pred, y_data)
    x_phys   = x_data.detach().clone().requires_grad_(True)
    grads    = torch.autograd.grad(
        model(x_phys).sum(), x_phys, create_graph=True
    )[0]
    l_mono   = torch.relu(-grads).pow(2).mean()
    l_bound  = torch.relu(0.6 - soh_pred).pow(2).mean()
    total    = l_data + lm * l_mono + lb * l_bound
    return total, {"total": total.item(), "data": l_data.item(),
                   "mono": l_mono.item()}


# ── 학습 ──────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    X_tr, y_tr, X_te, y_te, scaler, test_df = prepare_data()

    X_t = torch.tensor(X_tr, device=device)
    y_t = torch.tensor(y_tr, device=device).unsqueeze(1)

    model     = SOHCurvePINN(n_features=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=1e-5)

    best_loss = float("inf")
    history   = []

    print("학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss, bkd = compute_loss(model, X_t, y_t)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        history.append(bkd)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                "model":      model.state_dict(),
                "scaler":     scaler,
                "features":   INPUT_FEATURES,
                "nasa_rated": NASA_RATED,
                "note":       "Universal model - normalize by rated_cap before inference",
            }, SAVE_PATH)

        if epoch % 500 == 0:
            print(f"  [{epoch:5d}] total={bkd['total']:.5f}  "
                  f"data={bkd['data']:.5f}  mono={bkd['mono']:.5f}")

    print(f"\nBest loss: {best_loss:.5f}  ->  {SAVE_PATH}")

    # ── 평가 ──────────────────────────────────────────────────────────────────
    ckpt = torch.load(SAVE_PATH, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X_te, device=device)).cpu().numpy().flatten()

    true_pct = y_te * 100
    pred_pct = pred * 100

    mae  = np.mean(np.abs(pred_pct - true_pct))
    rmse = np.sqrt(np.mean((pred_pct - true_pct) ** 2))
    print(f"\n[{TEST_CELL} 테스트 결과]")
    print(f"  MAE : {mae:.2f}%")
    print(f"  RMSE: {rmse:.2f}%")

    _print_scale_demo(scaler)
    _plot(history, test_df, pred_pct, mae, rmse)


def _print_scale_demo(scaler):
    """
    다른 배터리 크기에서도 동일하게 동작함을 보여주는 예시.
    """
    print("\n[Universal 동작 예시]")
    print("  배터리 크기가 달라도 정규화 후 동일 모델 사용:\n")

    examples = [
        ("18650 (2 Ah)",  2.0,  0.38),
        ("21700 (5 Ah)",  5.0,  0.95),
        ("E-bike (20 Ah)", 20.0, 3.80),
        ("EV pack (150 Ah)", 150.0, 28.5),
    ]
    print(f"  {'배터리':20s}  {'cap_ah':>8}  {'cap_ratio':>10}  {'같은 범위?':>10}")
    print("  " + "-" * 56)
    for name, rated, cap_ah in examples:
        cap_ratio = cap_ah / rated
        in_range  = "YES" if 0.10 <= cap_ratio <= 0.25 else "NO"
        print(f"  {name:20s}  {cap_ah:>7.2f}Ah  {cap_ratio:>10.4f}  {in_range:>10}")
    print(f"\n  cap_ratio 는 배터리 크기와 무관하게 ~0.19 (SOH=100%) 로 수렴")


def _plot(history, test_df, pred_pct, mae, rmse):
    true_pct = test_df["soh"].values * 100
    disc_idx = test_df["disc_idx"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 손실 곡선
    ep = range(1, len(history) + 1)
    axes[0].semilogy(ep, [h["data"] for h in history], label="Data")
    axes[0].semilogy(ep, [h["mono"] for h in history], label="Monotonicity")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 추이
    axes[1].plot(disc_idx, true_pct, "k-", lw=2, label="True SOH")
    axes[1].plot(disc_idx, pred_pct, "r--", lw=1.5, label="PINN Pred")
    axes[1].set_xlabel("Discharge cycle")
    axes[1].set_ylabel("SOH (%)")
    axes[1].set_title(f"B0018 Test  MAE={mae:.2f}%  RMSE={rmse:.2f}%")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Scatter
    rng = [min(true_pct.min(), pred_pct.min()) - 1,
           max(true_pct.max(), pred_pct.max()) + 1]
    axes[2].plot(rng, rng, "k--", lw=1)
    axes[2].scatter(true_pct, pred_pct, s=15, alpha=0.6, color="steelblue")
    axes[2].set_xlabel("True SOH (%)")
    axes[2].set_ylabel("Predicted SOH (%)")
    axes[2].set_title(f"Pred vs True  MAE={mae:.2f}%")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Universal SOH PINN (capacity-normalized features)\n"
                 "Train: B0005/B0006/B0007  |  Test: B0018",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("soh_universal_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nGraph saved -> soh_universal_evaluation.png")


# ── 추론 함수 (외부 모듈 적용 시) ─────────────────────────────────────────────

def predict_soh(cap_ah_100_80, cap_ah_80_60,
                dv_dah_100_80, dv_dah_80_60,
                rated_cap,
                model_path=SAVE_PATH):
    """
    임의 크기 배터리에서 SOH 예측.

    Parameters
    ----------
    cap_ah_100_80  : 100->80% SOC 구간 방전 용량 (Ah)
    cap_ah_80_60   : 80->60%  SOC 구간 방전 용량 (Ah)
    dv_dah_100_80  : 100->80% SOC 구간 전압 기울기 (V/Ah)
    dv_dah_80_60   : 80->60%  SOC 구간 전압 기울기 (V/Ah)
    rated_cap      : 배터리 정격 용량 (Ah) - 라벨에서 확인
    """
    ckpt   = torch.load(model_path, weights_only=False, map_location="cpu")
    model  = SOHCurvePINN(n_features=4)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scaler = ckpt["scaler"]

    # 정규화
    x = np.array([[
        cap_ah_100_80 / rated_cap,
        cap_ah_80_60  / rated_cap,
        dv_dah_100_80 * rated_cap,
        dv_dah_80_60  * rated_cap,
    ]], dtype=np.float32)

    x_n = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        soh = float(model(torch.tensor(x_n)).item())
    return soh


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 55)
    print("Universal SOH PINN - capacity-normalized features")
    print(f"NASA rated_cap = {NASA_RATED} Ah")
    print("=" * 55 + "\n")
    train()
