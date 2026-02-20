from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

model_D  = joblib.load(BASE_DIR / "model/RF_model_D.joblib")
model_ED = joblib.load(BASE_DIR / "model/RF_model_ED.joblib")
