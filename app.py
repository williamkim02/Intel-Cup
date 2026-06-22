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

ROOT             = os.path.dirname(os.path.abspath(__file__))
MODEL_UNIVERSAL  = os.path.join(ROOT, "models", "soh_universal_model.pth")   # 100->60% 전체
MODEL_SEGMENT    = os.path.join(ROOT, "models", "soh_segment_model.pth")     # 임의 10% 구간
MODEL_PARKING    = os.path.join(ROOT, "models", "parking_model.pkl")
CUTOFF_V         = 2.5


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
    ckpt  = torch.load(MODEL_SEGMENT, weights_only=False, map_location="cpu")
    model = SOHCurvePINN(n_features=3)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["scaler"]

@st.cache_resource
def load_parking_model():
    return joblib.load(MODEL_PARKING)


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


def predict_soh_segment(cap_ah, dv_dah, soc_hi, soc_lo, rated_cap):
    model, scaler = load_segment_model()
    soc_mid = (soc_hi + soc_lo) / 2.0
    x = np.array([[
        cap_ah / rated_cap,
        dv_dah * rated_cap,
        soc_mid,
    ]], dtype=np.float32)
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
    if soh_pct >= 85:
        return "Good", "#2ecc71"
    elif soh_pct >= 75:
        return "Warning", "#f39c12"
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
        st.success("Battery is in good condition (SOH >= 85%). No action needed.")
    elif label == "Warning":
        st.warning("Battery health is declining (75~85%). Monitor closely.")
    else:
        st.error("Battery should be replaced (SOH < 75%). Risk of failure.")


# ── 페이지 ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Battery Diagnostic Module", page_icon="🔋", layout="wide")
st.title("🔋 Battery Diagnostic Module")
st.caption("Universal SOH measurement & Cold-weather discharge prevention")

tab1, tab2, tab3 = st.tabs(["SOH — Full Cycle", "SOH — Quick Segment", "Cold Weather Risk"])


# ════════════════════════════════════════════════════════════
# TAB 1 : SOH — 전체 구간 (100->60%)
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("SOH — Full Cycle (100% → 60% SOC)")
    st.caption("Higher accuracy | MAE 0.33% | Requires 100->60% discharge log")

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
            render_soh_result(soh * 100, rated_cap_f, "Universal PINN (MAE 0.33%)")
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
    st.subheader("SOH — Quick Segment (Any 10% SOC window)")
    st.caption("Quick measurement | MAE ~1.1% | Works with ANY 10% discharge window (e.g. 85->75%)")

    c1, c2 = st.columns([1, 1])
    with c1:
        rated_cap_s = st.number_input(
            "Rated Capacity (Ah)", min_value=0.1, max_value=5000.0,
            value=2.0, step=0.1, key="rc_seg",
        )
    with c2:
        st.info(
            "어떤 SOC 구간이든 가능합니다.  \n"
            "외부 모듈: 정전류 방전 5~10분 → 자동 계산  \n"
            "OBD-II: 주행 중 10% SOC 구간 감지 시 추출"
        )

    st.divider()
    mode_s = st.radio("Input method", ["Upload CSV", "Enter manually"], horizontal=True, key="mode_s")

    seg_input = None  # {cap_ah, dv_dah, soc_hi, soc_lo}

    if mode_s == "Upload CSV":
        st.markdown("필수 컬럼: `time`(s)  `voltage`(V)  `current`(A)")
        up_s = st.file_uploader("Upload discharge log", type=["csv"], key="up_seg")
        if up_s:
            df_raw_s = pd.read_csv(up_s)
            _, seg_feats, err = extract_features_from_csv(df_raw_s, rated_cap_s)
            if err:
                st.error(err)
            elif not seg_feats:
                st.warning("유효한 10% 구간 없음.")
            else:
                st.markdown("**감지된 SOC 구간 — 예측할 구간 선택:**")
                seg_df = pd.DataFrame(seg_feats)
                seg_df["label"] = seg_df.apply(
                    lambda r: f"SOC {int(r.soc_hi*100)}%→{int(r.soc_lo*100)}%  "
                              f"cap={r.cap_ah:.3f}Ah  dv={r.dv_dah:.3f}",
                    axis=1
                )
                chosen = st.selectbox("구간 선택", seg_df["label"].tolist())
                row = seg_feats[seg_df["label"].tolist().index(chosen)]
                seg_input = row
                with st.expander("Selected segment features"):
                    st.code(
                        f"SOC {int(row['soc_hi']*100)}%->{ int(row['soc_lo']*100)}%\n"
                        f"  cap_ah = {row['cap_ah']:.4f} Ah\n"
                        f"  dv_dah = {row['dv_dah']:.4f} V/Ah\n"
                        f"  soc_mid = {row['soc_mid']:.2f}"
                    )
    else:
        st.markdown("측정한 SOC 구간과 값을 입력하세요.")
        sc1, sc2 = st.columns(2)
        with sc1:
            soc_hi_in = st.slider("Segment start SOC (%)", 20, 100, 85, step=5) / 100
            soc_lo_in = st.slider("Segment end SOC (%)",   10,  95, 75, step=5) / 100
        with sc2:
            cap_s  = st.number_input("Discharged cap_ah (Ah)", key="cs",
                                      min_value=0.001, value=round(rated_cap_s*0.10, 3),
                                      step=0.001, format="%.3f")
            dv_s   = st.number_input("dV/dAh (V/Ah)", key="ds",
                                      max_value=0.0, value=-0.45, step=0.01, format="%.3f")

        if soc_hi_in <= soc_lo_in:
            st.error("Start SOC must be higher than End SOC.")
        else:
            seg_input = {"cap_ah": cap_s, "dv_dah": dv_s,
                         "soc_hi": soc_hi_in, "soc_lo": soc_lo_in}
            st.caption(f"Segment: {int(soc_hi_in*100)}%→{int(soc_lo_in*100)}%  "
                       f"soc_mid={((soc_hi_in+soc_lo_in)/2*100):.0f}%")

    st.divider()
    if st.button("Predict SOH", type="primary", use_container_width=True,
                 disabled=(seg_input is None), key="btn_seg"):
        try:
            soh = predict_soh_segment(
                seg_input["cap_ah"], seg_input["dv_dah"],
                seg_input["soc_hi"], seg_input["soc_lo"],
                rated_cap_s
            )
            render_soh_result(soh * 100, rated_cap_s, "Segment PINN (MAE ~1.1%)")
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
