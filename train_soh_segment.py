"""
train_soh_segment.py
임의 SOC 구간에서 SOH 예측하는 Segment PINN

핵심 아이디어:
  기존: 100->80%, 80->60% 고정 구간 필요
  개선: 어느 10% 구간이든 (예: 85->75%) SOH 예측 가능

추가 feature:
  soc_mid = 측정 구간의 중앙 SOC  (예: 85->75% -> 0.80)
  -> 모델이 "이 값이 곡선 어디서 나왔는지" 파악

입력 feature (3개):
  cap_ratio = cap_ah / rated_cap   (무차원, ~0.10)
  dv_norm   = dv_dah * rated_cap   (V)
  soc_mid   = 구간 중앙 SOC (0~1)

슬라이딩 윈도우:
  각 방전 곡선에서 10% 크기 윈도우를 5% 간격으로 슬라이딩
  -> 사이클당 ~16개 샘플 -> 풍부한 학습 데이터

학습: B0005/B0006/B0007   테스트: B0018
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
from train_soh_curves import SOHCurvePINN

# ── 설정 ──────────────────────────────────────────────────────────────────────
MAT_DIR    = os.path.join(os.path.dirname(__file__), "../5.+Battery+Data+Set")
CSV_DIR    = os.path.join(os.path.dirname(__file__), "../NASA")

TRAIN_CELLS = ["B0005", "B0006", "B0007"]
TEST_CELL   = "B0018"
NASA_RATED  = 2.0        # Ah

WINDOW_SIZE = 0.10       # SOC 구간 크기 (10%)
STRIDE      = 0.05       # 슬라이딩 간격 (5%)
SOC_TOL     = 0.01       # 구간 경계 허용 오차

INPUT_FEATURES = ["cap_ratio", "dv_norm", "soc_mid"]

N_NOISE   = 20
NOISE_STD = {
    "cap_ratio": 0.0005,   # 전류 센서 오차
    "dv_norm":   0.015,    # 전압 측정 오차
    "soc_mid":   0.010,    # BMS SOC 추정 오차
}

LAMBDA_MONO  = 0.3
LAMBDA_BOUND = 0.1
EPOCHS  = 4000
LR      = 1e-3
SAVE_PATH = "soh_segment_model.pth"


# ── 슬라이딩 윈도우 feature 추출 ──────────────────────────────────────────────

def sliding_windows(V, I, T, cap_total, rated_cap):
    """
    하나의 방전 곡선에서 슬라이딩 윈도우로 feature 추출.

    반환: list of dict  {cap_ratio, dv_norm, soc_mid}
    """
    dt    = np.diff(T, prepend=T[0])
    cumAh = np.cumsum(np.abs(I) * dt) / 3600.0
    soc   = np.clip(1.0 - cumAh / max(cap_total, 1e-6), 0.0, 1.0)

    # 윈도우 중앙 SOC 목록: 0.95, 0.90, ..., 0.25
    centers = np.arange(1.0 - WINDOW_SIZE / 2,
                        WINDOW_SIZE / 2 - STRIDE,
                        -STRIDE)

    samples = []
    for center in centers:
        hi = center + WINDOW_SIZE / 2
        lo = center - WINDOW_SIZE / 2
        if lo < 0.05:   # 너무 낮은 구간 제외 (노이즈 많음)
            break

        mask = (soc <= hi + SOC_TOL) & (soc >= lo - SOC_TOL)
        if mask.sum() < 4:
            continue

        seg_Ah = cumAh[mask]
        seg_V  = V[mask]
        cap_ah = seg_Ah[-1] - seg_Ah[0]
        if cap_ah < 1e-6:
            continue
        dv_dah = (seg_V[-1] - seg_V[0]) / cap_ah

        samples.append({
            "cap_ratio": cap_ah / rated_cap,
            "dv_norm":   dv_dah * rated_cap,
            "soc_mid":   float(center),
        })

    return samples


def extract_all_windows(mat_path, cell_name, gt_csv_path, rated_cap):
    """
    셀 전체 방전 사이클 -> 슬라이딩 윈도우 feature DataFrame.
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

        data      = c["data"][0, 0]
        V         = data["Voltage_measured"].flatten().astype(np.float64)
        I         = data["Current_measured"].flatten().astype(np.float64)
        T         = data["Time"].flatten().astype(np.float64)
        cap_total = float(data["Capacity"].flatten()[0])

        windows = sliding_windows(V, I, T, cap_total, rated_cap)
        for w in windows:
            rows.append({
                "cell":    cell_name,
                "disc_idx": disc_no,
                "soh":     soh_pct / 100.0,
                **w,
            })

    return pd.DataFrame(rows)


# ── 데이터 준비 ───────────────────────────────────────────────────────────────

def prepare_data():
    rng = np.random.default_rng(42)

    print("슬라이딩 윈도우 feature 추출 중...")
    train_frames = []
    for cell in TRAIN_CELLS:
        df = extract_all_windows(
            os.path.join(MAT_DIR, f"{cell}.mat"), cell,
            os.path.join(CSV_DIR, f"plecs_params_{cell}.csv"), NASA_RATED,
        )
        print(f"  {cell}: {len(df):,} 샘플  "
              f"({df['disc_idx'].nunique()} 사이클 x ~{len(df)//df['disc_idx'].nunique():.0f} 윈도우)")
        train_frames.append(df)

    test_df = extract_all_windows(
        os.path.join(MAT_DIR, f"{TEST_CELL}.mat"), TEST_CELL,
        os.path.join(CSV_DIR, f"plecs_params_{TEST_CELL}.csv"), NASA_RATED,
    )
    print(f"  {TEST_CELL}: {len(test_df):,} 샘플  (테스트)")

    train_df = pd.concat(train_frames, ignore_index=True)

    print(f"\n[Feature 범위]")
    for f in INPUT_FEATURES:
        print(f"  {f}: {train_df[f].min():.4f} ~ {train_df[f].max():.4f}")

    X_raw = train_df[INPUT_FEATURES].values.astype(np.float32)
    y_raw = train_df["soh"].values.astype(np.float32)

    # 노이즈 증강
    X_list, y_list = [X_raw], [y_raw]
    for _ in range(N_NOISE):
        noisy = X_raw.copy()
        for j, f in enumerate(INPUT_FEATURES):
            noisy[:, j] += rng.normal(0, NOISE_STD[f], size=len(X_raw))
        X_list.append(noisy)
        y_list.append(y_raw)

    X_train = np.vstack(X_list).astype(np.float32)
    y_train = np.concatenate(y_list).astype(np.float32)
    print(f"\n학습 샘플: {len(X_train):,}  ({len(X_raw):,} 원본 x {N_NOISE+1} 버전)")

    scaler    = MinMaxScaler()
    X_train_n = scaler.fit_transform(X_train).astype(np.float32)
    X_test_n  = scaler.transform(
        test_df[INPUT_FEATURES].values.astype(np.float32)
    ).astype(np.float32)

    return X_train_n, y_train, X_test_n, test_df, scaler


# ── 손실 함수 ─────────────────────────────────────────────────────────────────

def compute_loss(model, x_data, y_data):
    soh_pred = model(x_data)
    l_data   = nn.functional.mse_loss(soh_pred, y_data)

    x_phys   = x_data.detach().clone().requires_grad_(True)
    grads    = torch.autograd.grad(
        model(x_phys).sum(), x_phys, create_graph=True
    )[0]
    l_mono   = torch.relu(-grads[:, :2]).pow(2).mean()  # cap_ratio, dv_norm 만 단조성
    l_bound  = torch.relu(0.6 - soh_pred).pow(2).mean()

    total = l_data + LAMBDA_MONO * l_mono + LAMBDA_BOUND * l_bound
    return total, {"total": total.item(), "data": l_data.item(), "mono": l_mono.item()}


# ── 학습 ──────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    X_tr, y_tr, X_te, test_df, scaler = prepare_data()

    X_t = torch.tensor(X_tr, device=device)
    y_t = torch.tensor(y_tr, device=device).unsqueeze(1)

    model     = SOHCurvePINN(n_features=3).to(device)   # 3 features
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=1e-5)

    best_loss = float("inf")
    history   = []

    print("\n학습 시작...")
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
                "window_size": WINDOW_SIZE,
                "note":       "Segment model - any 10% SOC window + soc_mid feature",
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

    true_pct = test_df["soh"].values * 100
    pred_pct = pred * 100

    mae_all  = np.mean(np.abs(pred_pct - true_pct))
    rmse_all = np.sqrt(np.mean((pred_pct - true_pct) ** 2))

    print(f"\n[{TEST_CELL} 전체 윈도우 결과]")
    print(f"  MAE : {mae_all:.2f}%")
    print(f"  RMSE: {rmse_all:.2f}%")

    # 구간별 성능
    print(f"\n[SOC 구간별 MAE]")
    bins = [(0.90, 1.00), (0.80, 0.90), (0.70, 0.80),
            (0.60, 0.70), (0.50, 0.60), (0.40, 0.50)]
    for lo, hi in bins:
        mask = (test_df["soc_mid"] >= lo) & (test_df["soc_mid"] < hi)
        if mask.sum() == 0:
            continue
        m = np.mean(np.abs(pred_pct[mask] - true_pct[mask]))
        print(f"  SOC {int(lo*100):3d}~{int(hi*100):3d}%: "
              f"MAE={m:.2f}%  (n={mask.sum()})")

    _plot(history, test_df, pred_pct, mae_all, rmse_all)
    return model, scaler


# ── 시각화 ────────────────────────────────────────────────────────────────────

def _plot(history, test_df, pred_pct, mae, rmse):
    true_pct = test_df["soh"].values * 100
    soc_mid  = test_df["soc_mid"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 손실 곡선
    ep = range(1, len(history) + 1)
    axes[0].semilogy(ep, [h["data"] for h in history], label="Data")
    axes[0].semilogy(ep, [h["mono"] for h in history], label="Monotonicity")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 구간별 scatter (색상 = soc_mid)
    sc = axes[1].scatter(true_pct, pred_pct, c=soc_mid, cmap="RdYlGn",
                         s=8, alpha=0.5, vmin=0.4, vmax=1.0)
    rng = [min(true_pct.min(), pred_pct.min()) - 1,
           max(true_pct.max(), pred_pct.max()) + 1]
    axes[1].plot(rng, rng, "k--", lw=1)
    plt.colorbar(sc, ax=axes[1], label="SOC mid")
    axes[1].set_xlabel("True SOH (%)"); axes[1].set_ylabel("Pred SOH (%)")
    axes[1].set_title(f"Pred vs True  MAE={mae:.2f}%")
    axes[1].grid(True, alpha=0.3)

    # SOC 구간별 MAE 바 차트
    bins     = [(0.90, 1.00), (0.80, 0.90), (0.70, 0.80),
                (0.60, 0.70), (0.50, 0.60), (0.40, 0.50)]
    labels   = [f"{int(lo*100)}-{int(hi*100)}%" for lo, hi in bins]
    mae_vals = []
    for lo, hi in bins:
        mask = (test_df["soc_mid"].values >= lo) & (test_df["soc_mid"].values < hi)
        m = np.mean(np.abs(pred_pct[mask] - true_pct[mask])) if mask.sum() > 0 else 0
        mae_vals.append(m)

    axes[2].barh(labels, mae_vals, color="steelblue", alpha=0.8)
    axes[2].axvline(1.0, color="red", linestyle="--", lw=1, label="1% threshold")
    axes[2].set_xlabel("MAE (%)")
    axes[2].set_title("MAE by SOC segment")
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis="x")

    plt.suptitle("Segment SOH PINN - Any 10% SOC window\n"
                 "Train: B0005/B0006/B0007  |  Test: B0018",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("soh_segment_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Graph saved -> soh_segment_evaluation.png")


# ── 추론 함수 ─────────────────────────────────────────────────────────────────

def predict_soh_segment(cap_ah, dv_dah, soc_hi, soc_lo, rated_cap,
                        model_path=SAVE_PATH):
    """
    임의 SOC 구간 측정값으로 SOH 예측.

    Parameters
    ----------
    cap_ah    : 측정 구간에서 방전된 Ah
    dv_dah    : 측정 구간의 dV/dAh (V/Ah, 음수)
    soc_hi    : 구간 시작 SOC (예: 0.85)
    soc_lo    : 구간 종료 SOC (예: 0.75)
    rated_cap : 배터리 정격 용량 (Ah)
    """
    ckpt   = torch.load(model_path, weights_only=False, map_location="cpu")
    model  = SOHCurvePINN(n_features=3)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scaler = ckpt["scaler"]

    soc_mid = (soc_hi + soc_lo) / 2.0
    x = np.array([[
        cap_ah / rated_cap,
        dv_dah * rated_cap,
        soc_mid,
    ]], dtype=np.float32)

    x_n = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        soh = float(model(torch.tensor(x_n)).item())
    return soh


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 55)
    print("Segment SOH PINN - Any 10% SOC window")
    print(f"Window={int(WINDOW_SIZE*100)}%  Stride={int(STRIDE*100)}%  rated_cap={NASA_RATED}Ah")
    print("=" * 55)
    train()

    # 추론 예시
    print("\n[추론 예시]")
    examples = [
        (0.38, -1.13, 1.00, 0.90, "100->90%"),
        (0.20, -0.45, 0.85, 0.75, "85->75%"),
        (0.19, -0.38, 0.70, 0.60, "70->60%"),
    ]
    for cap, dv, hi, lo, label in examples:
        soh = predict_soh_segment(cap, dv, hi, lo, NASA_RATED)
        print(f"  {label}: cap={cap}Ah  dv/dAh={dv}  -> SOH={soh*100:.1f}%")
