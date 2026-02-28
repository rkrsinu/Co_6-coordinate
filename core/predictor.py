from pathlib import Path
import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Load models only once (fast for Streamlit)
# --------------------------------------------------
@st.cache_resource
def load_models():

    model_D = joblib.load(BASE_DIR / "model/RF_model_D.joblib")
    model_ED = joblib.load(BASE_DIR / "model/RF_model_ED.joblib")
    sign_model = joblib.load(BASE_DIR / "model/RF_sign_model.joblib")

    return model_D, model_ED, sign_model


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict(features: np.ndarray):

    model_D, model_ED, sign_model = load_models()

    features = np.array(features, dtype=np.float32).reshape(1, -1)

    # 🔹 predict sign first
    sign = sign_model.predict(features)[0]

    # 🔹 predict |D|
    D_mag = model_D.predict(features)[0]

    # apply sign
    D = D_mag * sign

    # 🔹 predict E/D
    ED = model_ED.predict(features)[0]

    return D, ED
