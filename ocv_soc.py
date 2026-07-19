# -*- coding: utf-8 -*-
"""
ocv_soc.py — estimate SOC from discharge voltage (OCV→SOC lookup).

Lets the segment model score a SHORT partial-discharge window without needing the
full discharge curve or any capacity/rated-capacity value: the window's SOC_mid is
read off a reference voltage↔SOC curve instead of from position in a full sweep.

Reference curve `models/ocv_soc_reference.csv` is the mean 2 A CC discharge voltage vs
SOC position of the NASA 18650 cells, built from HEALTHY cycles only (SOH > 90) — an
OCV↔SOC curve is essentially a fresh-cell chemistry property, and using aged cycles
biases the voltage low and inflates the SOC estimate.

Caveat: the discharge-voltage↔SOC map still shifts with cell health, C-rate and
temperature, so this is an APPROXIMATE SOC (a soft positional feature), not a calibrated
fuel gauge. On the real DK-2500 cell it recovers SOH within ~1 %p of the position-based
value (93 vs 94 %) even though per-window SOC_mid can be ~0.2 off — SOC_mid only tells
the model where on the curve a window sits, so this accuracy is sufficient.
"""
import os
import numpy as np
import pandas as pd

_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "models", "ocv_soc_reference.csv")
_V = None   # voltage grid (ascending)
_S = None   # matching SOC grid (ascending)


def _load():
    global _V, _S
    if _V is None:
        t = pd.read_csv(_REF_PATH).sort_values("voltage")
        _V = t["voltage"].to_numpy(float)
        _S = t["soc"].to_numpy(float)
    return _V, _S


def voltage_to_soc(voltage):
    """Estimate SOC (0.0 empty … 1.0 full) from a discharge terminal voltage.
    Scalar or array. Clamped to [0, 1]; flat ends of the curve map to 0 / 1."""
    V, S = _load()
    soc = np.interp(voltage, V, S)          # V ascending → S ascending
    return float(np.clip(soc, 0.0, 1.0)) if np.isscalar(voltage) else np.clip(soc, 0.0, 1.0)
