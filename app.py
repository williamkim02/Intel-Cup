"""
app.py  -  Battery Diagnostic Module
SOH 측정 + 겨울철 방전 위험 예측 + RUL / Knee 분석

실행: D:/python3.12/python -m streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_model import SOHCurvePINN, RULPredictor
from knee_detector import KneeDetector

ROOT             = os.path.dirname(os.path.abspath(__file__))
MODEL_UNIVERSAL  = os.path.join(ROOT, "models", "soh_universal_model.pth")   # 100->60% 전체
MODEL_SEGMENT    = os.path.join(ROOT, "models", "soh_segment_model.pth")     # 임의 10% 구간
MODEL_PARKING    = os.path.join(ROOT, "models", "parking_model.pkl")
MODEL_RUL        = os.path.join(ROOT, "models", "rul_model.pth")
CUTOFF_V         = 2.5
EOL_SOH          = 80.0   # RUL 기준 SOH (%)


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_universal_model():
    ckpt  = torch.load(MODEL_UNIVERSAL, weights_only=False, map_location="cpu")
    model = SOHCurvePINN(n_features=4)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["scaler"]

@st.cache_resource
def load_segment_model():
    # Report-consistent deployed model: leakage-free voltage-window segment PINN
    # features = [V_start, V_end, dV, SOC_mid]  (NO capacity)  — final report §3.4.4
    ckpt  = torch.load(MODEL_SEGMENT, weights_only=False, map_location="cpu")
    arch  = ckpt.get("arch", {"n_features": 4, "hidden_dim": 128, "hidden_layers": 3})
    model = SOHCurvePINN(n_features=arch["n_features"],
                         hidden_dim=arch["hidden_dim"],
                         hidden_layers=arch["hidden_layers"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["scaler"]

@st.cache_resource
def load_parking_model():
    return joblib.load(MODEL_PARKING)

@st.cache_resource
def load_rul_model():
    ckpt  = torch.load(MODEL_RUL, weights_only=False, map_location="cpu")
    model = RULPredictor(n_features=3, rul_max=ckpt["rul_max"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["scaler"], ckpt["rul_max"]


# ── 예측 함수 ─────────────────────────────────────────────────────────────────

def predict_soh_universal(cap_100_80, cap_80_60, dv_100_80, dv_80_60, rated_cap):
    model, scaler = load_universal_model()
    x = np.array([[
        cap_100_80 / rated_cap,
        cap_80_60  / rated_cap,
        dv_100_80  * rated_cap,
        dv_80_60   * rated_cap,
    ]], dtype=np.float32)
    x_n = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        return float(model(torch.tensor(x_n)).item())


def predict_soh_segment(v_start, v_end, soc_mid):
    """Report-consistent segment PINN: 4 voltage-only window features, NO capacity.
    features = [V_start, V_end, dV, SOC_mid].  soc_mid: 1.0=start of discharge, 0.0=end."""
    model, scaler = load_segment_model()
    x = np.array([[
        v_start,
        v_end,
        v_start - v_end,   # dV
        soc_mid,
    ]], dtype=np.float32)
    x_n = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        return float(model(torch.tensor(x_n)).item())


# ── Report-consistent segment feature extraction (voltage-window, sliding) ────

N_TS_SEG   = 128    # resample discharge to 128 steps (matches training)
WIN_SEG    = 13     # 13/128 ≈ 10% window
STRIDE_SEG = 6      # 6/128  ≈ 5% stride


def extract_segment_windows(df):
    """V(t)/I(t) discharge CSV -> sliding 10% voltage windows.
    Returns list of dicts: {soc_mid, v_start, v_end, dv}.  No capacity used."""
    df = df.copy().sort_values("time").reset_index(drop=True)
    V  = df["voltage"].values.astype(float)
    if len(V) < WIN_SEG + 1:
        return []
    # resample voltage over the discharge span to 128 steps (time-normalised)
    Vr = np.interp(np.linspace(0, 1, N_TS_SEG),
                   np.linspace(0, 1, len(V)), V)
    wins = []
    for ws in range(0, N_TS_SEG - WIN_SEG + 1, STRIDE_SEG):
        we      = ws + WIN_SEG
        soc_mid = 1.0 - (ws + we) / 2.0 / N_TS_SEG   # 1.0=start, 0.0=end
        v_s     = float(Vr[ws]); v_e = float(Vr[we - 1])
        wins.append({
            "soc_mid": round(soc_mid, 3),
            "v_start": round(v_s, 4),
            "v_end":   round(v_e, 4),
            "dv":      round(v_s - v_e, 4),
        })
    return wins


def predict_rul(soh_pct, rate, post_knee_flag):
    model, scaler, rul_max = load_rul_model()
    x   = np.array([[soh_pct, rate, float(post_knee_flag)]], dtype=np.float32)
    x_n = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        return float(model(torch.tensor(x_n)).item())


def predict_startup(park_hours, init_soc_pct, amb_temp_c):
    model = load_parking_model()
    v = float(model.predict([[park_hours, init_soc_pct, amb_temp_c]])[0])
    return v, v >= CUTOFF_V


def extract_features_from_csv(df, rated_cap):
    """V(t)/I(t) CSV -> 전 구간 feature + 슬라이딩 윈도우 feature."""
    df = df.copy().sort_values("time").reset_index(drop=True)

    if "capacity_ah" in df.columns:
        cumAh     = df["capacity_ah"].values
        cap_total = cumAh[-1]
    else:
        dt        = np.diff(df["time"].values, prepend=df["time"].values[0])
        cumAh     = np.cumsum(np.abs(df["current"].values) * dt) / 3600.0
        cap_total = cumAh[-1]

    if cap_total <= 0:
        return None, None, "capacity_ah 가 0 입니다."

    soc = np.clip(1.0 - cumAh / cap_total, 0.0, 1.0)
    V   = df["voltage"].values

    # 전체 구간 feature (100->80%, 80->60%)
    full_feats = {}
    for hi, lo in [(1.00, 0.80), (0.80, 0.60)]:
        label = f"{int(hi*100)}_{int(lo*100)}"
        mask  = (soc <= hi + 0.02) & (soc >= lo - 0.02)
        if mask.sum() < 5:
            full_feats = None
            break
        seg_Ah = cumAh[mask]
        seg_V  = V[mask]
        cap_ah = seg_Ah[-1] - seg_Ah[0]
        dv_dah = (seg_V[-1] - seg_V[0]) / cap_ah if cap_ah > 1e-6 else np.nan
        full_feats[f"cap_{label}"] = cap_ah
        full_feats[f"dv_{label}"]  = dv_dah

    # 슬라이딩 윈도우 feature (어느 구간이든)
    centers    = np.arange(0.95, 0.10, -0.05)
    seg_feats  = []
    for center in centers:
        hi   = center + 0.05
        lo   = center - 0.05
        mask = (soc <= hi + 0.01) & (soc >= lo - 0.01)
        if mask.sum() < 4:
            continue
        seg_Ah = cumAh[mask]
        seg_V  = V[mask]
        cap_ah = seg_Ah[-1] - seg_Ah[0]
        if cap_ah < 1e-6:
            continue
        dv_dah = (seg_V[-1] - seg_V[0]) / cap_ah
        seg_feats.append({
            "soc_hi":  round(hi, 2),
            "soc_lo":  round(lo, 2),
            "soc_mid": round(center, 2),
            "cap_ah":  round(cap_ah, 4),
            "dv_dah":  round(dv_dah, 4),
        })

    return full_feats, seg_feats, None


def soh_status(soh_pct):
    # Final report three-class decision: Good > 80, Marginal 70-80, Replace < 70
    if soh_pct > 80:
        return "Good", "#2ecc71"
    elif soh_pct >= 70:
        return "Marginal", "#f39c12"
    else:
        return "Replace", "#e74c3c"


def render_soh_result(soh_pct, rated_cap, model_name=""):
    label, color = soh_status(soh_pct)
    c1, c2, c3 = st.columns(3)
    c1.metric("SOH", f"{soh_pct:.1f}%")
    c2.metric("Status", label)
    c3.metric("Remaining Capacity",
              f"{soh_pct/100 * rated_cap:.2f} Ah  /  {rated_cap:.1f} Ah")
    if model_name:
        st.caption(f"Model: {model_name}")
    st.markdown(
        f"""
        <div style="background:#ddd;border-radius:8px;height:32px;width:100%;margin:8px 0">
          <div style="background:{color};border-radius:8px;height:32px;
                      width:{min(soh_pct,100):.1f}%;display:flex;align-items:center;
                      padding-left:12px;color:white;font-weight:bold;font-size:15px">
            {soh_pct:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if label == "Good":
        st.success("Battery is in good condition (SOH > 80%). Suitable for reuse.")
    elif label == "Marginal":
        st.warning("Marginal health (70–80%). Additional screening recommended before second-life use.")
    else:
        st.error("Battery should be replaced (SOH < 70%). Not suitable for reuse.")


# ── 페이지 ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Battery Diagnostic Module", page_icon="🔋", layout="wide")
st.title("🔋 Battery Diagnostic Module")
st.caption("Universal SOH measurement & Cold-weather discharge prevention")

tab1, tab2, tab3, tab4 = st.tabs([
    "SOH — Full Cycle",
    "SOH — Quick Segment",
    "Cold Weather Risk",
    "Lifetime Tracking",
])


# ════════════════════════════════════════════════════════════
# TAB 1 : SOH — 전체 구간 (100->60%)
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("SOH — Full Cycle (100% → 60% SOC)")
    st.caption("Full-discharge PINN (capacity-informed reference) | R²=0.944±0.005 on B0018 "
               "| Requires 100→60% discharge log")

    c1, c2 = st.columns([1, 1])
    with c1:
        rated_cap_f = st.number_input(
            "Rated Capacity (Ah)", min_value=0.1, max_value=5000.0,
            value=2.0, step=0.1, key="rc_full",
            help="배터리 라벨 Ah 값. kWh 팩: 에너지(Wh) / 팩전압(V)"
        )
    with c2:
        st.info("kWh -> Ah:  Energy(Wh) / Pack Voltage(V)  \n예) 75,000 Wh / 400 V = **187.5 Ah**")

    st.divider()
    mode_f = st.radio("Input method", ["Upload CSV", "Enter manually"], horizontal=True, key="mode_f")

    cap_100_80 = cap_80_60 = dv_100_80 = dv_80_60 = None

    if mode_f == "Upload CSV":
        st.markdown("필수 컬럼: `time`(s)  `voltage`(V)  `current`(A)")
        up = st.file_uploader("Upload discharge log", type=["csv"], key="up_full")
        if up:
            df_raw = pd.read_csv(up)
            st.dataframe(df_raw.head(5), use_container_width=True)
            full_f, seg_f, err = extract_features_from_csv(df_raw, rated_cap_f)
            if err:
                st.error(err)
            elif full_f is None:
                st.warning("100->60% 전체 구간 데이터가 없습니다. Quick Segment 탭을 사용하세요.")
            else:
                cap_100_80 = full_f["cap_100_80"]
                cap_80_60  = full_f["cap_80_60"]
                dv_100_80  = full_f["dv_100_80"]
                dv_80_60   = full_f["dv_80_60"]
                with st.expander("Extracted features"):
                    st.code(
                        f"100->80%  cap={cap_100_80:.4f}Ah  dv/dAh={dv_100_80:.4f} V/Ah\n"
                        f"80->60%   cap={cap_80_60:.4f}Ah   dv/dAh={dv_80_60:.4f} V/Ah"
                    )
    else:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**100% → 80% SOC**")
            cap_100_80 = st.number_input("cap_ah (Ah)", key="c1f", min_value=0.001,
                                          value=round(rated_cap_f*0.19, 3), step=0.001, format="%.3f")
            dv_100_80  = st.number_input("dV/dAh (V/Ah)", key="d1f", max_value=0.0,
                                          value=-1.13, step=0.01, format="%.3f")
        with cb:
            st.markdown("**80% → 60% SOC**")
            cap_80_60  = st.number_input("cap_ah (Ah)", key="c2f", min_value=0.001,
                                          value=round(rated_cap_f*0.19, 3), step=0.001, format="%.3f")
            dv_80_60   = st.number_input("dV/dAh (V/Ah)", key="d2f", max_value=0.0,
                                          value=-0.68, step=0.01, format="%.3f")

    st.divider()
    ready_f = all(v is not None for v in [cap_100_80, cap_80_60, dv_100_80, dv_80_60])
    if st.button("Predict SOH", type="primary", use_container_width=True,
                 disabled=not ready_f, key="btn_full"):
        try:
            soh = predict_soh_universal(cap_100_80, cap_80_60, dv_100_80, dv_80_60, rated_cap_f)
            render_soh_result(soh * 100, rated_cap_f, "Full-discharge PINN (R²=0.944, capacity-informed reference)")
        except Exception as e:
            st.error(f"Error: {e}")

    with st.expander("How to extract features (Python code)"):
        st.code("""
dt    = np.diff(T, prepend=T[0])
cumAh = np.cumsum(np.abs(I) * dt) / 3600.0
soc   = 1.0 - cumAh / cap_total_ah

for hi, lo in [(1.0, 0.8), (0.8, 0.6)]:
    mask   = (soc <= hi+0.02) & (soc >= lo-0.02)
    cap_ah = cumAh[mask][-1] - cumAh[mask][0]
    dv_dah = (V[mask][-1] - V[mask][0]) / cap_ah
""", language="python")


# ════════════════════════════════════════════════════════════
# TAB 2 : SOH — 빠른 구간 측정 (임의 10%)
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("SOH — Quick Segment (Any 10% discharge window)")
    st.caption("Deployed model | Leakage-free voltage-window PINN [V_start, V_end, ΔV, SOC_mid] "
               "| per-cycle R²=0.921, per-window R²=0.895±0.013 (B0018) | NO capacity input")

    c1, c2 = st.columns([1, 1])
    with c1:
        rated_cap_s = st.number_input(
            "Rated Capacity (Ah)  —  display only, not used by the model", min_value=0.1,
            max_value=5000.0, value=2.0, step=0.1, key="rc_seg",
        )
    with c2:
        st.info(
            "The deployed segment PINN uses **voltage-only** window features.  \n"
            "Rated capacity is used **only** to show remaining Ah — it does **not** enter the prediction."
        )

    st.divider()
    mode_s = st.radio("Input method", ["Upload CSV", "Enter manually"], horizontal=True, key="mode_s")

    seg_input = None   # {v_start, v_end, soc_mid}
    seg_all   = None   # list of all windows (for per-cycle averaging)

    if mode_s == "Upload CSV":
        st.markdown("Required columns: `time`(s)  `voltage`(V)  `current`(A)")
        up_s = st.file_uploader("Upload discharge log", type=["csv"], key="up_seg")
        if up_s:
            df_raw_s = pd.read_csv(up_s)
            wins = extract_segment_windows(df_raw_s)
            if not wins:
                st.warning("Not enough samples to form a 10% window.")
            else:
                seg_all = wins
                st.markdown("**Sliding 10% windows detected — pick one, or use the per-cycle average below:**")
                wdf = pd.DataFrame(wins)
                wdf["label"] = wdf.apply(
                    lambda r: f"window @ SOC_mid {r.soc_mid:.2f}   "
                              f"V {r.v_start:.3f}→{r.v_end:.3f} V   ΔV {r.dv:.3f}",
                    axis=1
                )
                chosen = st.selectbox("Window", wdf["label"].tolist())
                seg_input = wins[wdf["label"].tolist().index(chosen)]
                with st.expander("Selected window features (model input)"):
                    st.code(
                        f"V_start = {seg_input['v_start']:.4f} V\n"
                        f"V_end   = {seg_input['v_end']:.4f} V\n"
                        f"dV      = {seg_input['dv']:.4f} V\n"
                        f"SOC_mid = {seg_input['soc_mid']:.3f}   (1.0=start of discharge, 0.0=end)"
                    )
    else:
        st.markdown("Enter one 10% discharge window (voltage-only).")
        sc1, sc2 = st.columns(2)
        with sc1:
            v_start_in = st.number_input("V_start (V)", key="vs_in", min_value=0.0, max_value=5.0,
                                          value=3.90, step=0.01, format="%.3f")
            v_end_in   = st.number_input("V_end (V)",   key="ve_in", min_value=0.0, max_value=5.0,
                                          value=3.80, step=0.01, format="%.3f")
        with sc2:
            soc_mid_in = st.slider("Window position SOC_mid (1.0=start → 0.0=end)",
                                    0.0, 1.0, 0.5, step=0.05, key="socmid_in")
        if v_end_in > v_start_in:
            st.error("V_end should be ≤ V_start on a discharge window.")
        else:
            seg_input = {"v_start": v_start_in, "v_end": v_end_in,
                         "dv": v_start_in - v_end_in, "soc_mid": soc_mid_in}
            st.caption(f"ΔV = {v_start_in - v_end_in:.3f} V   SOC_mid = {soc_mid_in:.2f}")

    st.divider()

    # per-cycle average over all windows (this is the R²=0.921 metric in the report)
    if seg_all:
        if st.button("Predict SOH — per-cycle (mean of all windows)", type="primary",
                     use_container_width=True, key="btn_seg_cycle"):
            try:
                preds = [predict_soh_segment(w["v_start"], w["v_end"], w["soc_mid"]) * 100
                         for w in seg_all]
                soh_mean = float(np.mean(preds))
                render_soh_result(soh_mean, rated_cap_s,
                                  f"Segment PINN — per-cycle mean of {len(preds)} windows")
                st.caption(f"Per-window spread: {np.min(preds):.1f}–{np.max(preds):.1f}%  "
                           f"(std {np.std(preds):.1f}%p). Averaging windows is what gives the reported per-cycle accuracy.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Predict SOH — this single window", use_container_width=True,
                 disabled=(seg_input is None), key="btn_seg"):
        try:
            soh = predict_soh_segment(seg_input["v_start"], seg_input["v_end"], seg_input["soc_mid"])
            render_soh_result(soh * 100, rated_cap_s, "Segment PINN (single 10% window)")
        except Exception as e:
            st.error(f"Error: {e}")


# ════════════════════════════════════════════════════════════
# TAB 3 : 겨울철 방전 위험
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Cold Weather Discharge Risk")
    st.markdown("주차 후 시동이 걸릴지 예측합니다.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        park_hours = st.slider("Parking duration (hours)", 0, 48, 12)
    with c2:
        init_soc   = st.slider("SOC at parking (%)", 10, 100, 80, step=5)
    with c3:
        amb_temp   = st.slider("Expected temperature (°C)", -20, 10, -10)

    st.divider()

    if st.button("Check Startup Risk", type="primary", use_container_width=True):
        try:
            v_start, will_start = predict_startup(park_hours, init_soc, amb_temp)
            margin = v_start - CUTOFF_V

            c1, c2 = st.columns(2)
            c1.metric("Predicted Startup Voltage", f"{v_start:.3f} V",
                      delta=f"{margin:+.3f} V vs {CUTOFF_V}V cutoff",
                      delta_color="normal" if will_start else "inverse")
            c2.metric("Engine Start", "YES" if will_start else "NO")

            if will_start:
                if margin > 0.3:
                    st.success(f"Safe. {v_start:.2f}V is well above {CUTOFF_V}V cutoff.")
                else:
                    st.warning(f"Marginal — {margin:.3f}V above cutoff. Consider charging first.")
            else:
                st.error(f"HIGH RISK: {v_start:.2f}V is BELOW {CUTOFF_V}V cutoff.  \n"
                         "Charge the battery or use a battery warmer.")

            with st.expander(f"Full risk matrix at {amb_temp}°C"):
                pm    = load_parking_model()
                rows  = []
                hours = [0, 1, 2, 4, 8, 12, 24, 48]
                for s in [40, 60, 80, 100]:
                    row = {"SOC": f"{s}%"}
                    for h in hours:
                        v = float(pm.predict([[h, s, amb_temp]])[0])
                        row[f"{h}h"] = f"{'OK' if v>=CUTOFF_V else 'FAIL'} ({v:.2f}V)"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).set_index("SOC"), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

    st.caption("Valid: SOC 40-100% | Temp -20~0°C | Park 0-48h | PyBaMM NMC simulation")


# ════════════════════════════════════════════════════════════
# TAB 4 : Lifetime Tracking — Knee + RUL
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Lifetime Tracking — Knee Detection & RUL")
    st.caption(
        "SOH 측정 히스토리를 입력하면 Knee 감지 + 남은 수명(RUL) 예측  |  "
        "EOL 기준: SOH 80%  |  RUL MAE ~7 cycles"
    )

    with st.expander("Knee란?", expanded=False):
        st.markdown("""
**Knee** = 배터리 열화 곡선의 변곡점.

- **Knee 이전**: SEI 성장 지배 → 완만한 선형 열화
- **Knee 이후**: 리튬 플레이팅 시작 → 급속 열화 (양성 피드백 루프)

Knee를 넘은 배터리는 같은 SOH여도 잔여 수명이 훨씬 짧다.

| 상태 | 의미 |
|---|---|
| SOH > 80%, Knee 없음 | 정상 |
| SOH 60~80%, Knee 없음 | 2nd-life 재활용 가능 |
| Knee 감지됨 | 급속 열화 진입, 즉시 교체 검토 |
        """)

    st.divider()
    input_mode = st.radio(
        "SOH 히스토리 입력 방법",
        ["직접 입력 (쉼표 구분)", "CSV 업로드"],
        horizontal=True,
        key="rul_mode",
    )

    soh_list = []

    if input_mode == "직접 입력 (쉼표 구분)":
        st.markdown("측정 순서대로 SOH (%) 입력  예) `92.1, 91.3, 90.5, 89.2, ...`")
        raw = st.text_area("SOH 히스토리", value="", height=80, key="rul_text",
                           placeholder="92.1, 91.3, 90.5, 89.2, 88.1, 87.0, 85.8, 84.3, 82.9, 81.2, 79.5, 78.0")
        if raw.strip():
            try:
                soh_list = [float(v.strip()) for v in raw.replace("\n", ",").split(",") if v.strip()]
            except ValueError:
                st.error("숫자만 입력하세요.")
    else:
        st.markdown("필수 컬럼: `soh` (%)  옵션: `cycle` (사이클 번호)")
        up_r = st.file_uploader("CSV 업로드", type=["csv"], key="up_rul")
        if up_r:
            df_r = pd.read_csv(up_r)
            if "soh" not in df_r.columns:
                st.error("'soh' 컬럼이 없습니다.")
            else:
                soh_list = df_r["soh"].dropna().tolist()
                st.dataframe(df_r.head(5), use_container_width=True)

    if len(soh_list) < 5:
        st.info("최소 5회 이상의 SOH 측정값이 필요합니다.")
    else:
        # Knee 감지
        kd = KneeDetector.from_list(soh_list)
        info = kd.summary()
        rate      = info["rate_%/cyc"]
        post_knee = info["post_knee_flag"]

        st.divider()

        # ── 메트릭 ─────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("측정 사이클 수", f"{info['n_cycles']}")
        m2.metric("현재 SOH", f"{info['current_soh']:.1f}%")
        m3.metric("열화 속도", f"{rate:.3f} %/cycle")
        if info["knee_detected"]:
            m4.metric("Knee", f"감지 (cycle {info['knee_cycle']})",
                      delta=f"{info['cycles_since_knee']} cycles 전", delta_color="inverse")
        else:
            m4.metric("Knee", "미감지")

        # ── RUL 예측 ───────────────────────────────────────────
        current_soh = info["current_soh"]
        if current_soh < EOL_SOH:
            rul_cycles = 0
            st.warning(f"SOH가 이미 EOL 기준({EOL_SOH}%) 이하입니다. RUL = 0")
        else:
            rul_cycles = predict_rul(current_soh, rate, post_knee)
            rul_cycles = max(0.0, rul_cycles)

        c_rul1, c_rul2 = st.columns(2)
        c_rul1.metric("예상 잔여 수명 (RUL)", f"약 {rul_cycles:.0f} 사이클",
                      help="SOH 80% 도달까지 남은 방전 사이클 수 (MAE ~7 cycles)")

        # 권고 판정
        if info["knee_detected"]:
            rec_label  = "즉시 교체 권고"
            rec_color  = "error"
            rec_detail = (f"Knee 감지 (cycle {info['knee_cycle']}) — 급속 열화 진입. "
                          f"재활용 가치 낮음, 폐기 검토.")
        elif current_soh < EOL_SOH:
            rec_label  = "교체 필요"
            rec_color  = "error"
            rec_detail = f"SOH {current_soh:.1f}% — EOL({EOL_SOH}%) 도달."
        elif current_soh < 85 and rul_cycles < 15:
            rec_label  = "교체 임박"
            rec_color  = "warning"
            rec_detail = f"잔여 수명 약 {rul_cycles:.0f} 사이클. 조기 교체 준비 권장."
        elif current_soh < EOL_SOH + 10:
            rec_label  = "2nd-life 재활용 검토"
            rec_color  = "warning"
            rec_detail = "SOH 60~80% 구간 — 정지형 ESS 등 2nd-life 배터리로 재활용 가능."
        else:
            rec_label  = "정상"
            rec_color  = "success"
            rec_detail = f"SOH {current_soh:.1f}% — Knee 없음, 정상 운용 중."

        c_rul2.metric("판정", rec_label)

        if rec_color == "error":
            st.error(rec_detail)
        elif rec_color == "warning":
            st.warning(rec_detail)
        else:
            st.success(rec_detail)

        # ── SOH 히스토리 + RUL 예상 궤적 차트 ────────────────────
        st.divider()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Battery Lifetime Analysis", fontsize=12)

        cycs = np.arange(len(soh_list))

        # SOH history
        axes[0].plot(cycs, soh_list, "b-o", ms=3, lw=1.5, label="SOH history")
        axes[0].axhline(EOL_SOH, color="red",    linestyle="--", alpha=0.7, label=f"EOL ({EOL_SOH}%)")
        axes[0].axhline(85,      color="orange",  linestyle="--", alpha=0.5, label="Warning (85%)")
        if info["knee_detected"]:
            axes[0].axvline(info["knee_cycle"], color="purple", linestyle=":",
                            lw=1.5, label=f"Knee (cycle {info['knee_cycle']})")
        axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("SOH (%)")
        axes[0].set_title("SOH History"); axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # RUL projection
        proj_cycs = np.arange(0, max(int(rul_cycles) + 10, 20))
        proj_soh  = current_soh + rate * proj_cycs
        proj_soh  = np.clip(proj_soh, 0, 100)
        axes[1].plot(len(soh_list) - 1 + proj_cycs, proj_soh,
                     "r--", lw=1.5, label="Projected (linear)")
        axes[1].plot(cycs, soh_list, "b-", lw=1.5, label="History")
        axes[1].axhline(EOL_SOH, color="red", linestyle="--", alpha=0.7,
                        label=f"EOL ({EOL_SOH}%)")
        if rul_cycles > 0:
            eol_x = len(soh_list) - 1 + rul_cycles
            axes[1].axvline(eol_x, color="red", linestyle=":", alpha=0.6,
                            label=f"~EOL cycle {eol_x:.0f}")
        axes[1].set_xlabel("Cycle"); axes[1].set_ylabel("SOH (%)")
        axes[1].set_title(f"RUL Projection  (RUL~{rul_cycles:.0f} cycles)")
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "RUL: NASA B0005/B0006/B0007 학습, B0018 검증 (Leave-One-Out)  |  "
            "MAE ~7 cycles  |  Knee: 2차 미분 알고리즘 (ML 불필요)"
        )
