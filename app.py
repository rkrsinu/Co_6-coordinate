import streamlit as st

from core.xyz_parser import parse_xyz
from core.geometry import compute_geometry
from core.feature_builder import build_features
from core.predictor import predict
from utils.sorting import sort_bl_angles


st.set_page_config(layout="wide")
st.title("Co–Octahedral SIM → D & E/D Predictor")

uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])

if uploaded_file:

    try:
        elements, coords = parse_xyz(uploaded_file)

        bl, angles, idev = compute_geometry(elements, coords)

        bl, angles = sort_bl_angles(bl, angles)

        features = build_features(bl, angles, idev)

        D, ED = predict(features)

        st.success(f"D  = {D:.2f} cm⁻¹")
        st.success(f"E/D = {ED:.3f}")

        with st.expander("Show extracted geometry"):
            st.write("Bond lengths:", bl)
            st.write("Bond angles :", angles)
            st.write("Ideal deviation:", idev)

    except Exception as e:
        st.error(f"Error: {e}")
