# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:34:06 2025

@author: IDB
"""

# streamlit_app_ci_calibrated.py
import glob
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import CategoricalDtype

st.set_page_config(
    page_title="Payment Acceptance (Calibrated) + Confidence Interval",
    page_icon="💳",
    layout="centered"
)

CAL_MODELS_DIR = "bootstrap_models_cal"  # carpeta con modelos calibrados .pkl
FEATURE_ORDER = ["MidName", "Bin", "Application2", "Amount"]  # ¡mismo orden que en entrenamiento!

# ----------------- Load artifacts -----------------
@st.cache_data
def load_options():
    with open("app_options.json", "r", encoding="utf-8") as f:
        opts = json.load(f)
    # Normaliza tipos
    opts["MidName"] = [str(x) for x in opts["MidName"]]
    opts["BIN"] = [str(x) for x in opts["BIN"]]  # ojo: clave en JSON es "BIN"
    opts["Application2"] = [str(x) for x in opts["Application2"]]
    opts["Amount"] = [float(x) for x in opts["Amount"]]
    return opts

@st.cache_resource
def load_calibrated_models():
    paths = sorted(glob.glob(f"{CAL_MODELS_DIR}/lgbm_bootstrap_cal_*.pkl"))
    models = [joblib.load(p) for p in paths]
    return models, paths

opts = load_options()
models, model_paths = load_calibrated_models()

st.title("💳 Calibrated Payment Acceptance Probability")
st.caption("Mean calibrated probability with 95% confidence interval (bootstrap of calibrated LightGBM models)")

if not models:
    st.error(f"No calibrated models found in '{CAL_MODELS_DIR}/'. Run the calibrated bootstrap trainer first.")
    st.stop()

# ----------------- UI -----------------
col1, col2 = st.columns(2)
with col1:
    mid = st.selectbox("Merchant (MidName)", options=opts["MidName"], index=0)
    app2 = st.selectbox("Application2 (transaction type)", options=opts["Application2"], index=0)
with col2:
    bin_ = st.selectbox("BIN (Issuer BIN)", options=opts["BIN"], index=0)
    default_amt_idx = opts["Amount"].index(9.99) if 9.99 in opts["Amount"] else 0
    amount = st.selectbox("Amount", options=opts["Amount"], index=default_amt_idx, format_func=lambda x: f"{x:.2f}")

with st.sidebar:
    st.header("Ensemble settings")
    n_models_to_use = st.slider("Number of models to use", min_value=10, max_value=len(models), value=min(50, len(models)), step=10)
    st.write(f"Loaded models: **{len(models)}**")
    st.caption(f"Reading from: `{CAL_MODELS_DIR}/`")

st.markdown("---")

# ----------------- Build input row (MATCH TRAIN ORDER) -----------------
MID_CATS  = CategoricalDtype(categories=opts["MidName"])
BIN_CATS  = CategoricalDtype(categories=opts["BIN"])
APP2_CATS = CategoricalDtype(categories=opts["Application2"])

X_input = pd.DataFrame([{
    "MidName": str(mid).strip(),
    "Bin": str(bin_).strip(),              # ¡ojo: columna se llama 'Bin'!
    "Application2": str(app2).strip(),
    "Amount": float(amount)
}], columns=FEATURE_ORDER)

# Tipos categóricos
X_input["MidName"] = X_input["MidName"].astype(MID_CATS)
X_input["Bin"] = X_input["Bin"].astype(BIN_CATS)
X_input["Application2"] = X_input["Application2"].astype(APP2_CATS)
X_input["Amount"] = pd.to_numeric(X_input["Amount"], errors="coerce").fillna(0.0)

# ----------------- Debug panel -----------------
with st.expander("🔎 Debug (categories & codes)"):
    st.write("**Input row** (with categorical dtypes):")
    st.dataframe(X_input, use_container_width=True)
    info = {}
    for c in ["MidName", "Bin", "Application2"]:
        val = X_input[c].iloc[0]
        code = X_input[c].cat.codes.iloc[0]
        isna = pd.isna(val)
        info[c] = {"value": str(val), "code": int(code), "isna": bool(isna)}
    st.json(info)
    if any(v["code"] == -1 or v["isna"] for v in info.values()):
        st.warning("Some category is not in the training vocabulary (code = -1 or NaN). "
                   "Pick values from dropdowns or update app_options.json.")

# ----------------- Scoring -----------------
if st.button("Compute calibrated probability + 95% CI"):
    # Validación previa: no permitir NaN/-1
    bad = []
    for c in ["MidName", "Bin", "Application2"]:
        if X_input[c].isna().iloc[0] or X_input[c].cat.codes.iloc[0] == -1:
            bad.append(c)
    if bad:
        st.error(f"Invalid categories for: {', '.join(bad)}. Please choose values that exist in app_options.json.")
        st.stop()

    try:
        # Subconjunto del ensemble (para controlar tiempo)
        use_models = models[:n_models_to_use]

        # Cada .pkl es un CalibratedClassifierCV(prefit) → predict_proba calibrada
        probs = np.array([float(m.predict_proba(X_input)[:, 1][0]) for m in use_models])

        mean_p = float(probs.mean())
        lo, hi = np.percentile(probs, [2.5, 97.5])
        std = float(probs.std())

        st.metric("Calibrated acceptance probability (mean)", f"{mean_p*100:.2f}%")
        st.write(f"**95% confidence interval (bootstrap):** {lo*100:.2f}% – {hi*100:.2f}%")
        st.caption(f"Ensemble size used: {len(use_models)} | Std across models: {std:.4f}")

        with st.expander("Bootstrap distribution"):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(probs, bins=15)
            ax.axvline(lo, linestyle="--")
            ax.axvline(hi, linestyle="--")
            ax.set_xlabel("Calibrated predicted probability")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with st.expander("Input details"):
            st.dataframe(X_input, use_container_width=True)

        with st.expander("Model files (sample)"):
            for p in model_paths[:10]:
                st.text(p)
            if len(model_paths) > 10:
                st.text(f"... (+{len(model_paths)-10} more)")

    except Exception as e:
        st.error(f"Scoring error: {e}")
        st.info("Confirm model column names, order, and category vocabularies match the training pipeline.")

# ----------------- Footer -----------------
st.markdown("---")
st.caption("Calibrated with CalibratedClassifierCV (isotonic/sigmoid fallback). "
           "Confidence intervals computed via bootstrap percentiles across calibrated models.")
