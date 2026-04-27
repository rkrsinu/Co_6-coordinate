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

    # -------- D --------
    tree_preds_D = np.array([
        tree.predict(features)[0] for tree in model_D.estimators_
    ])

    D_mean = np.mean(tree_preds_D)
    D_std  = np.std(tree_preds_D)

    # -------- E/D --------
    tree_preds_ED = np.array([
        tree.predict(features)[0] for tree in model_ED.estimators_
    ])

    ED_mean = np.mean(tree_preds_ED)
    ED_std  = np.std(tree_preds_ED)

    return D_mean, D_std, ED_mean, ED_std
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
