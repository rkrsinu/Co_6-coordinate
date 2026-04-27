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

    with st.spinner("Processing..."):

        try:
            elements, coords = parse_xyz(uploaded_file)

            bl, angles, idev = compute_geometry(elements, coords)

            bl, angles = sort_bl_angles(bl, angles)

            features = build_features(bl, angles, idev)

            D, D_std, ED, ED_std = predict(features)

            # 🔥 Minimum uncertainty threshold
            D_unc = max(20, D_std)
            ED_unc = max(0.05, ED_std)

            st.success(f"D  = {D:.2f} ± {D_unc:.2f} cm⁻¹")
            st.success(f"E/D = {ED:.3f} ± {ED_unc:.3f}")

            # 🔥 warning for unreliable predictions
            if D_std > 50:
                st.warning("⚠️ High uncertainty: molecule may be outside training domain")

            with st.expander("Show extracted geometry"):
                st.write("Bond lengths:", bl)
                st.write("Bond angles :", angles)
                st.write("Ideal deviation:", idev)

        except Exception as e:
            st.error(f"Error: {e}")
