import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from inference import run_inference
from synthetic_gfs import generate_synthetic_operational_data
from config import Config
import os
import datetime
import altair as alt

st.set_page_config(page_title="MausamGuard | SIH 2026", layout="wide")

st.title("🌩️ MausamGuard: AI Forecast Bust Predictor")
st.markdown(f"**Operational Mode:** `{Config.MODE}` | Predicting when physical weather models (GFS) will fail.")

# --- Sidebar Controls ---
st.sidebar.header("Controls")
selected_date = st.sidebar.date_input("Forecast Initialization", datetime.date(2023, 7, 6))
lead_time = st.sidebar.slider("Lead Time (Days)", 1, 10, 5)

# --- Generate/Load Data ---
@st.cache_data
def get_inference_data():
    """Runs the ML inference pipeline. Uses Config.MODE to determine data source."""
    if Config.MODE == "DEMO":
        # Check if synthetic data exists, if not generate it
        if not os.path.exists(f"{Config.DATA_DIR}/synthetic_merged_data.parquet"):
            generate_synthetic_operational_data()
        
        # Load the synthetic raw data
        raw_df = pd.read_parquet(f"{Config.DATA_DIR}/synthetic_merged_data.parquet")
    else:
        # Load real data
        raw_df = pd.read_parquet(f"{Config.DATA_DIR}/real_merged_data.parquet")
    
    # Run the ML model (from inference.py)
    try:
        results = run_inference(raw_df)
        return results, raw_df
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

with st.spinner("Running AI Inference..."):
    results_df, raw_df = get_inference_data()

if not results_df.empty:
    # Filter for the selected date and lead time
    # Convert date to string or datetime to match
    mask = (pd.to_datetime(results_df['Date']).dt.date == selected_date) & (results_df['Lead_Time'] == lead_time)
    display_df = results_df[mask]
    
    if display_df.empty:
        st.warning(f"No data available for Lead Time Day {lead_time} on {selected_date}.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Confidence Map - Day {lead_time}")
            
            # Color scale
            display_df['color_r'] = (display_df['Bust_Probability'] * 255).astype(int)
            display_df['color_g'] = 0
            display_df['color_b'] = ((1 - display_df['Bust_Probability']) * 255).astype(int)
            
            layer = pdk.Layer(
                "ColumnLayer",
                display_df,
                get_position=["Lon", "Lat"],
                get_elevation="Bust_Probability",
                elevation_scale=100000,
                radius=20000,
                get_fill_color=["color_r", "color_g", "color_b", 150],
                pickable=True,
                auto_highlight=True,
            )
            
            view_state = pdk.ViewState(
                latitude=23.0,
                longitude=80.0,
                zoom=3.5,
                pitch=45,
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Lat: {Lat}, Lon: {Lon}\nRisk: {Risk_Level} ({Bust_Probability})"},
                map_style='mapbox://styles/mapbox/dark-v10',
            )
            st.pydeck_chart(r)
            
            st.markdown("---")
            st.subheader("Forecast vs Observation (Ground Truth Comparison)")
            # Show a time-series for the highest risk point
            max_risk = display_df.loc[display_df['Bust_Probability'].idxmax()]
            ts_mask = (raw_df['Lat'] == max_risk['Lat']) & (raw_df['Lon'] == max_risk['Lon']) & (raw_df['Lead_Time'] == lead_time)
            ts_data = raw_df[ts_mask].sort_values('Date')
            
            chart_data = ts_data[['Date', 'GFS_T2m', 'ERA5_T2m']].melt('Date', var_name='Source', value_name='Temperature (C)')
            chart = alt.Chart(chart_data).mark_line(point=True).encode(
                x='Date:T',
                y=alt.Y('Temperature (C):Q', scale=alt.Scale(zero=False)),
                color='Source:N',
                tooltip=['Date', 'Source', 'Temperature (C)']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
        with col2:
            st.subheader("Regional Analysis")
            st.info("Showing AI analysis for the highest risk region on this day.")
            
            st.error(f"⚠️ **Highest Risk Region Detected**")
            st.markdown(f"**Coordinates:** {max_risk['Lat']}°N, {max_risk['Lon']}°E")
            st.markdown(f"**Bust Probability:** {max_risk['Bust_Probability'] * 100:.1f}%")
            
            st.markdown("### AI Explainability")
            st.markdown("*Real SHAP values driving the prediction:*")
            
            # Normalize SHAP values for display (make them absolute percentages of total impact)
            shap_cols = ['SHAP_Z500_Grad', 'SHAP_Temporal_Delta', 'SHAP_Climatology']
            total_shap = abs(max_risk[shap_cols[0]]) + abs(max_risk[shap_cols[1]]) + abs(max_risk[shap_cols[2]]) + 0.0001
            
            grad_pct = abs(max_risk['SHAP_Z500_Grad']) / total_shap
            temp_pct = abs(max_risk['SHAP_Temporal_Delta']) / total_shap
            clim_pct = abs(max_risk['SHAP_Climatology']) / total_shap
            
            st.progress(float(grad_pct), text=f"Z500 Spatial Gradient Impact ({grad_pct*100:.0f}%)")
            st.progress(float(temp_pct), text=f"Temporal Inconsistency Impact ({temp_pct*100:.0f}%)")
            st.progress(float(clim_pct), text=f"Climatological Anomaly Impact ({clim_pct*100:.0f}%)")
            
            st.markdown("---")
            if max_risk['Bust_Probability'] > 0.7:
                st.markdown("**Recommendation:** Disaster Response forces in this region should prepare for scenarios outside the standard GFS forecast envelope.")
            else:
                st.markdown("**Recommendation:** GFS model is behaving normally. Standard operational procedures apply.")
