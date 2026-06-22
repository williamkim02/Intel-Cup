"""
app.py  -  Battery Diagnostic Module
SOH 측정 + 겨울철 방전 위험 예측

실행: D:/python3.12/python -m streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_model import SOHCurvePINN

ROOT            = os.path.dirname(os.path.abspath(__file__))
MODEL_SOH       = os.path.join(ROOT, "models", "soh_universal_model.pth")
MODEL_PARKING   = os.path.join(ROOT, "models", "parking_model.pkl")
CUTOFF_V        = 2.5


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_soh_model():
    ckpt   = torch.load(MODEL_SOH, weights_only=False, map_location="cpu")
    model  = SOHCurvePINN(n_features=4)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["scaler"]

@st.cache_resource
def load_parking_model():
    return joblib.load(MODEL_PARKING)


# ── 예측 함수 ─────────────────────────────────────────────────────────────────

def predict_soh(cap_100_80, cap_80_60, dv_100_80, dv_80_60, rated_cap):
    model, scaler = load_soh_model()
    x = np.array([[
        cap_100_80 / rated_cap,
        cap_80_60  / rated_cap,
        dv_100_80  * rated_cap,
        dv_80_60   * rated_cap,
    ]], dtype=np.float32)
    x_n = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        soh = float(model(torch.tensor(x_n)).item())
    return soh


def predict_startup(park_hours, init_soc_pct, amb_temp_c):
    model = load_parking_model()
    v = float(model.predict([[park_hours, init_soc_pct, amb_temp_c]])[0])
    return v, v >= CUTOFF_V


def extract_features_from_csv(df, rated_cap):
    """
    V(t)/I(t) CSV -> cap_ah, dv_dah per SOC interval.
    필수 컬럼: time(s), voltage(V), current(A), capacity_ah(Ah) or soc(0~1 or 0~100)
    """
    df = df.copy().sort_values("time").reset_index(drop=True)

    # 누적 Ah 계산
    if "capacity_ah" in df.columns:
        cumAh = df["capacity_ah"].values
        cap_total = cumAh[-1]
    else:
        dt    = np.diff(df["time"].values, prepend=df["time"].values[0])
        cumAh = np.cumsum(np.abs(df["current"].values) * dt) / 3600.0
        cap_total = cumAh[-1]

    if cap_total <= 0:
        return None, "capacity_ah 가 0 입니다. 데이터를 확인하세요."

    soc = np.clip(1.0 - cumAh / cap_total, 0.0, 1.0)
    V   = df["voltage"].values

    results = {}
    intervals = [(1.00, 0.80), (0.80, 0.60)]
    for hi, lo in intervals:
        label = f"{int(hi*100)}_{int(lo*100)}"
        mask  = (soc <= hi + 0.02) & (soc >= lo - 0.02)
        if mask.sum() < 5:
            return None, f"SOC {int(hi*100)}~{int(lo*100)}% 구간 데이터 부족 ({mask.sum()}점)"
        seg_Ah = cumAh[mask]
        seg_V  = V[mask]
        cap_ah = seg_Ah[-1] - seg_Ah[0]
        dv_dah = (seg_V[-1] - seg_V[0]) / cap_ah if cap_ah > 1e-6 else np.nan
        results[f"cap_{label}"] = cap_ah
        results[f"dv_{label}"]  = dv_dah

    return results, None


def soh_status(soh_pct):
    if soh_pct >= 85:
        return "Good", "#2ecc71"
    elif soh_pct >= 75:
        return "Warning", "#f39c12"
    else:
        return "Replace", "#e74c3c"


def render_soh_result(soh_pct, rated_cap):
    label, color = soh_status(soh_pct)
    col1, col2, col3 = st.columns(3)
    col1.metric("SOH", f"{soh_pct:.1f}%")
    col2.metric("Status", label)
    col3.metric("Remaining Capacity",
                f"{soh_pct/100 * rated_cap:.2f} Ah  /  {rated_cap:.1f} Ah")

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
        st.success("Battery is in good condition (SOH >= 85%). No action needed.")
    elif label == "Warning":
        st.warning("Battery health is declining (75~85%). Monitor closely.")
    else:
        st.error("Battery should be replaced (SOH < 75%). Risk of failure.")


# ── 페이지 설정 ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Battery Diagnostic Module",
    page_icon="🔋",
    layout="wide",
)

st.title("🔋 Battery Diagnostic Module")
st.caption("Universal SOH measurement & Cold-weather discharge prevention")

tab1, tab2 = st.tabs(["SOH Measurement", "Cold Weather Risk"])


# ════════════════════════════════════════════════════════════
# TAB 1 : SOH 측정
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("State of Health (SOH) Prediction")

    # ── 배터리 정격용량 ──
    st.markdown("#### 1. Battery Info")
    c1, c2 = st.columns([1, 1])
    with c1:
        rated_cap = st.number_input(
            "Rated Capacity (Ah)  — from battery label",
            min_value=0.1, max_value=5000.0,
            value=2.0, step=0.1,
            help="배터리 라벨의 Ah 값. kWh 팩은 에너지(Wh) ÷ 팩전압(V) 으로 변환"
        )
    with c2:
        st.info(
            "**kWh pack -> Ah**  \n"
            "Ah = Energy (Wh) / Pack Voltage (V)  \n"
            "예) 75,000 Wh / 400 V = **187.5 Ah**"
        )

    st.divider()

    # ── 입력 방식 선택 ──
    st.markdown("#### 2. Measurement Data")
    mode = st.radio(
        "Input method",
        ["Upload V(t)/I(t) CSV", "Enter values manually"],
        horizontal=True,
    )

    cap_100_80 = cap_80_60 = dv_100_80 = dv_80_60 = None

    # ── CSV 업로드 모드 ──
    if mode == "Upload V(t)/I(t) CSV":
        st.markdown(
            "CSV 필수 컬럼: **`time`**(s)  **`voltage`**(V)  **`current`**(A)  \n"
            "선택 컬럼: `capacity_ah`(Ah) — 없으면 전류 적분으로 자동 계산"
        )
        uploaded = st.file_uploader("Upload discharge log CSV", type=["csv"])

        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.dataframe(df_raw.head(5), use_container_width=True)

            feats, err = extract_features_from_csv(df_raw, rated_cap)
            if err:
                st.error(f"Feature 추출 실패: {err}")
            else:
                cap_100_80 = feats["cap_100_80"]
                cap_80_60  = feats["cap_80_60"]
                dv_100_80  = feats["dv_100_80"]
                dv_80_60   = feats["dv_80_60"]

                with st.expander("Extracted features"):
                    st.code(
                        f"100->80%  cap_ah={cap_100_80:.4f} Ah   dv/dAh={dv_100_80:.4f} V/Ah\n"
                        f"80->60%   cap_ah={cap_80_60:.4f} Ah    dv/dAh={dv_80_60:.4f} V/Ah"
                    )

    # ── 수동 입력 모드 ──
    else:
        st.markdown("100% → 60% SOC 방전 후 각 구간의 측정값을 입력하세요.")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**100% → 80% SOC**")
            cap_100_80 = st.number_input(
                "Discharged capacity (Ah)", key="c1",
                min_value=0.001, value=round(rated_cap * 0.19, 3), step=0.001, format="%.3f"
            )
            dv_100_80 = st.number_input(
                "Voltage slope dV/dAh (V/Ah)", key="d1",
                max_value=0.0, value=-1.13, step=0.01, format="%.3f"
            )
        with cb:
            st.markdown("**80% → 60% SOC**")
            cap_80_60 = st.number_input(
                "Discharged capacity (Ah)", key="c2",
                min_value=0.001, value=round(rated_cap * 0.19, 3), step=0.001, format="%.3f"
            )
            dv_80_60 = st.number_input(
                "Voltage slope dV/dAh (V/Ah)", key="d2",
                max_value=0.0, value=-0.68, step=0.01, format="%.3f"
            )

    st.divider()

    # ── 예측 ──
    ready = all(v is not None for v in [cap_100_80, cap_80_60, dv_100_80, dv_80_60])
    if st.button("Predict SOH", type="primary", use_container_width=True, disabled=not ready):
        try:
            soh_pct = predict_soh(cap_100_80, cap_80_60, dv_100_80, dv_80_60, rated_cap) * 100
            st.markdown("#### Result")
            render_soh_result(soh_pct, rated_cap)

            with st.expander("Normalized features (debug)"):
                st.code(
                    f"cap_ratio_100_80 = {cap_100_80/rated_cap:.4f}  (healthy ~0.20)\n"
                    f"cap_ratio_80_60  = {cap_80_60/rated_cap:.4f}  (healthy ~0.20)\n"
                    f"dv_norm_100_80   = {dv_100_80*rated_cap:.4f} V\n"
                    f"dv_norm_80_60    = {dv_80_60*rated_cap:.4f} V"
                )
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # ── 코드 예시 ──
    with st.expander("How to extract features from V(t)/I(t) log (Python)"):
        st.code("""
import numpy as np

# V, I, T : sensor arrays (1 Hz sampling)
dt    = np.diff(T, prepend=T[0])
cumAh = np.cumsum(np.abs(I) * dt) / 3600.0   # cumulative Ah
soc   = 1.0 - cumAh / cap_total_ah            # SOC (0~1)

for hi, lo in [(1.0, 0.8), (0.8, 0.6)]:
    mask   = (soc <= hi + 0.02) & (soc >= lo - 0.02)
    cap_ah = cumAh[mask][-1] - cumAh[mask][0]
    dv_dah = (V[mask][-1] - V[mask][0]) / cap_ah
""", language="python")


# ════════════════════════════════════════════════════════════
# TAB 2 : 겨울철 방전 위험
# ════════════════════════════════════════════════════════════
with tab2:
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

            st.markdown("#### Result")
            col1, col2 = st.columns(2)
            col1.metric(
                "Predicted Startup Voltage", f"{v_start:.3f} V",
                delta=f"{margin:+.3f} V vs {CUTOFF_V}V cutoff",
                delta_color="normal" if will_start else "inverse"
            )
            col2.metric("Engine Start", "YES ✓" if will_start else "NO ✗")

            if will_start:
                if margin > 0.3:
                    st.success(f"Safe to park. {v_start:.2f}V is well above the {CUTOFF_V}V cutoff.")
                else:
                    st.warning(f"Marginal — only {margin:.3f}V above cutoff. Consider charging before parking.")
            else:
                st.error(
                    f"HIGH RISK: {v_start:.2f}V is BELOW {CUTOFF_V}V cutoff.  \n"
                    "Charge the battery or use a battery warmer before parking."
                )

            # 리스크 매트릭스
            with st.expander(f"Full risk matrix at {amb_temp}°C"):
                soc_vals  = [40, 60, 80, 100]
                hour_vals = [0, 1, 2, 4, 8, 12, 24, 48]
                pm        = load_parking_model()
                rows = []
                for s in soc_vals:
                    row = {"SOC": f"{s}%"}
                    for h in hour_vals:
                        v = float(pm.predict([[h, s, amb_temp]])[0])
                        row[f"{h}h"] = f"{'OK' if v >= CUTOFF_V else 'FAIL'} ({v:.2f}V)"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).set_index("SOC"), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.caption(
        "Valid range: SOC 40-100% | Temperature -20~0°C | Parking 0-48h  \n"
        "Model trained on PyBaMM simulation (Chen2020, NMC chemistry)"
    )
