from pathlib import Path
import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Load models once
# --------------------------------------------------
@st.cache_resource
def load_models():
    model_D = joblib.load(BASE_DIR / "model/RF_model_D.joblib")
    model_ED = joblib.load(BASE_DIR / "model/RF_model_ED.joblib")
    return model_D, model_ED


# --------------------------------------------------
# Prediction with uncertainty
# --------------------------------------------------
def predict(features: np.ndarray):

    model_D, model_ED = load_models()

    features = np.array(features, dtype=np.float64).reshape(1, -1)

    # -------- D prediction --------
    tree_preds_D = []
    for tree in model_D.estimators_:
        tree_preds_D.append(tree.predict(features)[0])

    tree_preds_D = np.array(tree_preds_D)

    D_mean = np.mean(tree_preds_D)
    D_std  = np.std(tree_preds_D)

    # -------- E/D prediction --------
    tree_preds_ED = []
    for tree in model_ED.estimators_:
        tree_preds_ED.append(tree.predict(features)[0])

    tree_preds_ED = np.array(tree_preds_ED)

    ED_mean = np.mean(tree_preds_ED)
    ED_std  = np.std(tree_preds_ED)

    return D_mean, D_std, ED_mean, ED_std
