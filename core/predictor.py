import joblib
import numpy as np

# load once (fast for Streamlit)
model_D  = joblib.load("model/RF_model_D.joblib")
model_ED = joblib.load("model/RF_model_ED.joblib")

def predict(features: np.ndarray):

    features = features.reshape(1, -1)

    D  = model_D.predict(features)[0]
    ED = model_ED.predict(features)[0]

    return D, ED
