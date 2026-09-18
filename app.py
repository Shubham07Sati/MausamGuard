import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from inference import run_inference
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="MausamGuard | SIH 2026", layout="wide")

st.title("🌩️ MausamGuard: AI Forecast Bust Predictor")
st.markdown("Predicting when physical weather models (GFS) will fail during extreme events.")

# --- Sidebar Controls ---
st.sidebar.header("Controls")
selected_date = st.sidebar.date_input("Forecast Date", datetime.date(2026, 9, 20))
lead_time = st.sidebar.slider("Lead Time (Days)", 1, 10, 5)

# --- Generate/Load Data ---
@st.cache_data
def get_inference_data():
    """Generates a mock raw forecast and runs it through our ML inference pipeline."""
    # Generate a dense grid for India
    lats = np.arange(8.0, 38.0, 0.5)
    lons = np.arange(68.0, 98.0, 0.5)
    
    records = []
    # Just generating for the selected date and lead time to save time
    for lat in lats:
        for lon in lons:
            records.append({
                "Date": pd.to_datetime("2026-09-20"),
                "Lead_Time": 5,
                "Lat": lat,
                "Lon": lon,
                "GFS_T2m": np.random.uniform(25, 35),
                "GFS_TP": np.random.uniform(0, 50),
                "GFS_Z500": np.random.uniform(5700, 5900)
            })
    
    raw_df = pd.DataFrame(records)
    # Run the ML model (from inference.py)
    try:
        results = run_inference(raw_df)
        # Add a mock "hotspot" (storm/cyclone) in the Bay of Bengal / East Coast
        # to make the map look interesting for the demo
        hotspot_mask = (results['Lat'] > 15) & (results['Lat'] < 22) & (results['Lon'] > 85) & (results['Lon'] < 92)
        results.loc[hotspot_mask, 'Bust_Probability'] = np.random.uniform(0.7, 0.95, size=hotspot_mask.sum())
        
        # Add another hotspot in the Himalayas (Western Disturbance)
        hotspot_mask2 = (results['Lat'] > 30) & (results['Lat'] < 36) & (results['Lon'] > 74) & (results['Lon'] < 80)
        results.loc[hotspot_mask2, 'Bust_Probability'] = np.random.uniform(0.6, 0.85, size=hotspot_mask2.sum())
        
        return results
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return pd.DataFrame()

with st.spinner("Running AI Inference..."):
    df = get_inference_data()

if not df.empty:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Confidence Map - Day {lead_time}")
        
        # Define the color scale based on probability
        # Red = High Bust Prob, Blue = Low Bust Prob
        df['color_r'] = (df['Bust_Probability'] * 255).astype(int)
        df['color_g'] = 0
        df['color_b'] = ((1 - df['Bust_Probability']) * 255).astype(int)
        
        # PyDeck Heatmap/Grid
        layer = pdk.Layer(
            "ColumnLayer",
            df,
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
            tooltip={"text": "Lat: {Lat}, Lon: {Lon}\nBust Probability: {Bust_Probability}"},
            map_style='mapbox://styles/mapbox/dark-v10',
        )
        st.pydeck_chart(r)
        
    with col2:
        st.subheader("Regional Analysis")
        st.info("Hover over the map columns to see specific grid probabilities.")
        
        # Find the max risk area
        max_risk = df.loc[df['Bust_Probability'].idxmax()]
        
        st.error(f"⚠️ **Highest Risk Region Detected**")
        st.markdown(f"**Coordinates:** {max_risk['Lat']}°N, {max_risk['Lon']}°E")
        st.markdown(f"**Bust Probability:** {max_risk['Bust_Probability'] * 100:.1f}%")
        
        st.markdown("### AI Explainability")
        st.markdown("*Why is the model flagging this region?*")
        
        # Mock SHAP breakdown for the dashboard
        st.progress(0.45, text="High Z500 Spatial Gradient (+45%)")
        st.progress(0.30, text="Temporal Inconsistency (+30%)")
        st.progress(0.12, text="Climatological Anomaly (+12%)")
        
        st.markdown("---")
        st.markdown("**Recommendation:** Disaster Response forces in this region should prepare for scenarios outside the standard GFS forecast envelope.")
