import streamlit as st
import numpy as np

from core.xyz_parser import parse_xyz
from core.geometry import compute_geometry
from core.feature_builder import build_feature_vector
from core.predictor import load_model, predict

st.set_page_config(layout="wide")

st.title("Co–Octahedral SIM → D & E/D Predictor")

uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])

if uploaded_file:

    elements, coords = parse_xyz(uploaded_file)

    d, A, ideal_dev = compute_geometry(elements, coords)

    features = build_feature_vector(d, A, ideal_dev)

    st.write("### Sorted Bond Lengths")
    st.write(np.round(d, 3))

    st.write("### Sorted Bond Angles")
    st.write(np.round(A, 2))

    st.write("Ideal deviation:", round(ideal_dev, 3))

    model, model_type = load_model("model/model.pth")

    pred = predict(model, model_type, features)

    st.success(f"D  = {pred[0][0]:.2f}")
    st.success(f"E/D = {pred[0][1]:.3f}")
