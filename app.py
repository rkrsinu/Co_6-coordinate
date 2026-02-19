import streamlit as st
from core.xyz_parser import parse_xyz
from core.geometry import *
from core.feature_builder import build_features
from core.predictor import load_model, predict
from utils.sorting import sort_features

st.title("Co–Octahedral SIM Predictor")

uploaded_file = st.file_uploader("Upload XYZ", type=["xyz"])

if uploaded_file:

    elements, coords = parse_xyz(uploaded_file)

    co_idx = get_co_index(elements)
    neighbors = get_6_neighbors(elements, coords, co_idx)

    bond_lengths = calc_bond_lengths(coords, co_idx, neighbors)
    bond_angles = calc_angles(coords, co_idx, neighbors)

    bond_lengths, bond_angles = sort_features(bond_lengths, bond_angles)

    features = build_features(bond_lengths, bond_angles)

    model, model_type = load_model("model/model.pth")

    pred = predict(model, model_type, features)

    st.success(f"D = {pred[0][0]:.2f}")
    st.success(f"E/D = {pred[0][1]:.3f}")
