from pathlib import Path
import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Load models (no sign model anymore)
# --------------------------------------------------
@st.cache_resource
def load_models():

    model_D = joblib.load(BASE_DIR / "model/RF_model_D.joblib")
    model_ED = joblib.load(BASE_DIR / "model/RF_model_ED.joblib")

    return model_D, model_ED


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict(features: np.ndarray):

    model_D, model_ED = load_models()

    # 🔥 MUST match training dtype
    features = np.array(features, dtype=np.float64).reshape(1, -1)

    # Direct prediction (NO sign model)
    D = model_D.predict(features)[0]
    ED = model_ED.predict(features)[0]

    return D, ED
