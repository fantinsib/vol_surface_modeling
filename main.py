import streamlit as st
import pandas as pd
import tools as t
import tools_dupire as td

st.title("Volatility Surface Modeling")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'Ts' not in st.session_state:
    st.session_state.Ts = None
if 'models' not in st.session_state:
    st.session_state.models = None

use_synthetic = st.checkbox("Use synthetic demo data")

if use_synthetic:
    st.subheader("Synthetic Data Parameters")
    st.session_state.n_maturities = st.number_input("Number of maturities", 1, 500, 100)
    st.session_state.strikes_per_maturity = st.number_input("Number of strikes/maturity", 1, 200, 20)
    st.session_state.base_vol = st.number_input("Base volatility", 0.01, 1.0, 0.15, 0.01)
    st.session_state.noise = st.number_input("Noise level", 0.0, 1.0, 0.001, 0.001, format="%.4f")
    st.session_state.vol_effect = st.number_input("Volatility skew factor", 0.0, 5.0, 0.3, 0.1, format="%.4f")
    st.session_state.r = 0.01
    st.session_state.spot = 100


    if st.button("Generate synthetic data"):
        df = t.synthetic_options_dataset(
            n_maturities=st.session_state.n_maturities,
            strikes_per_maturity=st.session_state.strikes_per_maturity,
            spot=st.session_state.spot,
            r=st.session_state.r,
            base_vol=st.session_state.base_vol,
            noise=st.session_state.noise,
            vol_effect=st.session_state.vol_effect,
        )
        st.session_state.df = df
        st.success("Synthetic dataset generated!")
        st.dataframe(df)

else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("File uploaded")
        st.dataframe(df)


if st.session_state.df is not None:
    if st.button("Plot Volatility Surface"):
        df = st.session_state.df
        svi_params = td.svi_calibration(df)  
        interpolation_models = td.interpolation_svi(svi_params)
        st.session_state.Ts = df["Maturity"].unique()

        st.info("Displaying interactive volatility surface...")
        td.interactive_vol_surface(st.session_state.Ts, interpolation_models, data_points=df, spot=100,r=0.01)
    
    if st.button('Plot Dupire Local Volatility Surface'):
        df = st.session_state.df
        svi_params = td.svi_calibration(df)  
        interpolation_models = td.interpolation_svi(svi_params)
        st.session_state.Ts = df["Maturity"].unique()
        st.session_state.models = td.interpolation_svi(svi_params)
        st.session_state.vol_surface = td.volatility_surface(df, interpolation_models)
        st.info("Displaying interactive Dupire Local Volatility Surface...")
        td.display_dupire_surface(st.session_state.vol_surface, st.session_state.Ts, interpolation_models, k_max = 0.5)

