# streamlit_app_ci_calibrated.py
import os, glob, tarfile, json, joblib, numpy as np, pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from pandas.api.types import CategoricalDtype

st.set_page_config(
    page_title="Payment Acceptance (Calibrated) + Confidence Interval",
    page_icon="💳",
    layout="centered"
)

# === AJUSTA ESTO ===
REPO_ID = "ssoler90/payment-acceptance-models"     # <-- pon tu repo de HF Hub
# Opción A (simple): fija el nombre del tar subido a la Hub
MODEL_FILENAME = "bootstrap_models_cal_20251020.tar.gz"
# Opción B (recomendada): usa version.json en la Hub (ver más abajo)
CACHE_DIR = "hf_cache"
FEATURE_ORDER = ["MidName", "Bin", "Application2", "Amount"]

@st.cache_data
def load_version(repo_id: str):
    """Lee version.json en la Hub para saber qué tar descargar."""
    # Descomenta estas 4 líneas si usas version.json en tu repo de HF:
    # p = hf_hub_download(repo_id=repo_id, filename="version.json", cache_dir=CACHE_DIR,
    #                     token=st.secrets.get("HF_TOKEN", None))
    # with open(p, "r", encoding="utf-8") as f:
    #     return json.load(f)["model_filename"]
    return MODEL_FILENAME  # si no usas version.json, usa constante

@st.cache_data
def download_and_unpack(repo_id: str, filename: str, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    artifact_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        token=st.secrets.get("HF_TOKEN", None)  # usa secreto si el repo es privado
    )
    target_dir = os.path.join(cache_dir, "bootstrap_models_cal")
    if not os.path.isdir(target_dir):
        with tarfile.open(artifact_path, "r:gz") as t:
            t.extractall(cache_dir)
    return target_dir

@st.cache_data
def load_options(repo_id: str):
    local = hf_hub_download(
        repo_id=repo_id,
        filename="app_options.json",
        cache_dir=CACHE_DIR,
        token=st.secrets.get("HF_TOKEN", None)
    )
    with open(local, "r", encoding="utf-8") as f:
        opts = json.load(f)
    opts["MidName"]      = [str(x) for x in opts["MidName"]]
    opts["BIN"]          = [str(x) for x in opts["BIN"]]
    opts["Application2"] = [str(x) for x in opts["Application2"]]
    opts["Amount"]       = [float(x) for x in opts["Amount"]]
    return opts

@st.cache_resource
def load_calibrated_models(models_root: str):
    paths = sorted(glob.glob(os.path.join(models_root, "bootstrap_models_cal", "lgbm_bootstrap_cal_*.pkl")))
    models = [joblib.load(p) for p in paths]
    return models, paths

# ↓ Descarga artefacto + carga modelos y vocabularios
MODEL_FILENAME = load_version(REPO_ID)
MODELS_ROOT = download_and_unpack(REPO_ID, MODEL_FILENAME, CACHE_DIR)
OPTS = load_options(REPO_ID)
MODELS, MODEL_PATHS = load_calibrated_models(CACHE_DIR)

st.title("💳 Calibrated Payment Acceptance Probability")
st.caption("Mean calibrated probability with 95% CI (bootstrap of calibrated LightGBM models)")

if not MODELS:
    st.error("No calibrated models found after download. Check REPO_ID and artifact name / version.json.")
    st.stop()

# ----------------- UI -----------------
c1, c2 = st.columns(2)
with c1:
    mid  = st.selectbox("Merchant (MidName)", OPTS["MidName"], index=0)
    app2 = st.selectbox("Application2 (transaction type)", OPTS["Application2"], index=0)
with c2:
    bin_ = st.selectbox("BIN (Issuer BIN)", OPTS["BIN"], index=0)
    idx = OPTS["Amount"].index(9.99) if 9.99 in OPTS["Amount"] else 0
    amount = st.selectbox("Amount", OPTS["Amount"], index=idx, format_func=lambda x: f"{x:.2f}")

with st.sidebar:
    st.header("Ensemble settings")
    n_models_to_use = st.slider("Number of models to use", 10, len(MODELS), min(50, len(MODELS)), step=10)
    st.caption(f"Loaded models: {len(MODELS)}")

# ----------------- Build input row (typed) -----------------
MID_C  = CategoricalDtype(categories=OPTS["MidName"])
BIN_C  = CategoricalDtype(categories=OPTS["BIN"])
APP2_C = CategoricalDtype(categories=OPTS["Application2"])

X = pd.DataFrame([{
    "MidName": str(mid).strip(),
    "Bin": str(bin_).strip(),
    "Application2": str(app2).strip(),
    "Amount": float(amount)
}], columns=FEATURE_ORDER)

X["MidName"]      = X["MidName"].astype(MID_C)
X["Bin"]          = X["Bin"].astype(BIN_C)
X["Application2"] = X["Application2"].astype(APP2_C)
X["Amount"]       = pd.to_numeric(X["Amount"], errors="coerce").fillna(0.0)

with st.expander("🔎 Debug"):
    info = {}
    for c in ["MidName","Bin","Application2"]:
        info[c] = {"value": str(X[c].iloc[0]), "code": int(X[c].cat.codes.iloc[0]), "isna": bool(X[c].isna().iloc[0])}
    st.json(info)

# ----------------- Scoring -----------------
if st.button("Compute calibrated probability + 95% CI"):
    bad = [c for c in ["MidName","Bin","Application2"] if X[c].isna().iloc[0] or X[c].cat.codes.iloc[0] == -1]
    if bad:
        st.error(f"Unknown categories: {', '.join(bad)}. Update app_options.json in the HF repo.")
        st.stop()

    use = MODELS[:n_models_to_use]
    probs = np.array([float(m.predict_proba(X)[:,1][0]) for m in use])

    mean_p = float(probs.mean())
    lo, hi = np.percentile(probs, [2.5, 97.5])
    std = float(probs.std())

    st.metric("Calibrated acceptance probability (mean)", f"{mean_p*100:.2f}%")
    st.write(f"**95% CI:** {lo*100:.2f}% – {hi*100:.2f}%")
    st.caption(f"Models used: {len(use)} | Std: {std:.4f}")

    with st.expander("Bootstrap distribution"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(probs, bins=15)
        ax.axvline(lo, linestyle="--"); ax.axvline(hi, linestyle="--")
        ax.set_xlabel("Calibrated predicted probability"); ax.set_ylabel("Count")
        st.pyplot(fig)
