import streamlit as st

from core.xyz_parser import parse_xyz
from core.geometry import compute_geometry
from core.feature_builder import build_features
from core.predictor import predict
from utils.sorting import sort_bl_angles

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Co–Octahedral SIM → D & E/D Predictor")

uploaded_file = st.file_uploader("Upload XYZ file", type=["xyz"])

# --------------------------------------------------
# Main workflow
# --------------------------------------------------
if uploaded_file:

    with st.spinner("Processing..."):

        try:
            # -------- Parse XYZ --------
            elements, coords = parse_xyz(uploaded_file)

            # -------- Geometry --------
            bl, angles, idev = compute_geometry(elements, coords)

            # -------- Sorting --------
            bl, angles = sort_bl_angles(bl, angles)

            # -------- Features --------
            features = build_features(bl, angles, idev)

            # -------- Prediction --------
            D, D_std, ED, ED_std = predict(features)

            # --------------------------------------------------
            # 🔥 Adjust uncertainty (HALF of model std)
            # --------------------------------------------------
            D_unc = D_std / 2
            ED_unc = ED_std / 2

            # Optional lower bounds (recommended)
            if D_unc < 10:
                D_unc = 10
            if ED_unc < 0.02:
                ED_unc = 0.02

            # --------------------------------------------------
            # Display results
            # --------------------------------------------------
            st.success(f"D  = {D:.2f} ± {D_unc:.2f} cm⁻¹")
            st.success(f"E/D = {ED:.3f} ± {ED_unc:.3f}")

            # -------- Warning --------
            if D_std > 50:
                st.warning("⚠️ High uncertainty: molecule may be outside training domain")

            # -------- Show geometry --------
            with st.expander("Show extracted geometry"):
                st.write("Bond lengths:", bl)
                st.write("Bond angles :", angles)
                st.write("Ideal deviation:", idev)

        except Exception as e:
            st.error(f"Error: {e}")
