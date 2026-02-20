from pathlib import Path
import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent


@st.cache_resource
def load_models():

    model_D = joblib.load(BASE_DIR / "model/RF_model_D.joblib")
    model_ED = joblib.load(BASE_DIR / "model/RF_model_ED.joblib")

    return model_D, model_ED


def predict(features: np.ndarray):

    model_D, model_ED = load_models()

    features = features.reshape(1, -1)

    D = model_D.predict(features)[0]
    ED = model_ED.predict(features)[0]

    return D, ED
