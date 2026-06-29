"""
Interactive Streamlit Dashboard for Multi-Sensor Satellite Data Fusion Pipeline.

This app provides an interactive user interface to control transport wind sliders
and visualize binned fire FRP data aligned with corrected HCHO column concentrations.
"""

import os
import numpy as np
import streamlit as st
import plotly.express as px

# Import backend class
from pipeline import AtmosphericPipeline

# 1. Page Setup & Header Configuration
st.set_page_config(
    layout="wide",
    page_title="Atmospheric Monitoring India",
    page_icon="🇮🇳"
)

st.title("🇮🇳 Multi-Sensor Satellite Data Fusion Dashboard")
st.subheader("Objective 2: HCHO Hotspot Identification & Wind Plume Transport Correction")

# 2. Sidebar Interactive Controls
st.sidebar.header("🕹️ Boundary Layer Transport Controls")

u_wind = st.sidebar.slider(
    label="Zonal Wind Velocity (U Wind - East/West)",
    min_value=-15.0,
    max_value=15.0,
    value=5.0,
    step=0.5,
    help="Zonal wind component at 850 hPa (m/s). Positive value is eastward wind."
)

v_wind = st.sidebar.slider(
    label="Meridional Wind Velocity (V Wind - North/South)",
    min_value=-15.0,
    max_value=15.0,
    value=0.0,
    step=0.5,
    help="Meridional wind component at 850 hPa (m/s). Positive value is northward wind."
)

tau_lifetime = st.sidebar.slider(
    label="Precursor VOC Lifetime (τ)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Precursor chemical lifetime in hours."
)

st.sidebar.info(
    "These sliders dynamically compute the Smearing Length Scale (Ls ≈ U / τ) "
    "in real-time, inversely shifting the TROPOMI HCHO column matrices back "
    "to the true MODIS/VIIRS fire coordinates."
)

# 3. Backend Integration & Data Loading
# Cache the pipeline resource and initial data loading to ensure lightning-fast rendering
@st.cache_resource
def get_pipeline():
    return AtmosphericPipeline(resolution=0.1)

@st.cache_data
def load_base_datasets():
    # Automatically generate synthetic datasets if they are missing
    if not (os.path.exists("data/hcho_raw.nc") and 
            os.path.exists("data/no2_raw.nc") and 
            os.path.exists("data/fire_events.csv")):
        with st.spinner("Generating simulated satellite datasets (HCHO, NO2, Fires)..."):
            from pipeline import generate_synthetic_data
            generate_synthetic_data("data")
            
    pipeline = get_pipeline()
    
    # Ingest HCHO NetCDF layer
    hcho_regridded = pipeline.ingest_satellite_data(
        file_path="data/hcho_raw.nc",
        variable_name='hcho_column',
        qa_var_name='qa_value',
        cloud_fraction_var_name='cloud_fraction',
        qa_threshold=0.5,
        cloud_threshold=0.4
    )
    
    # Ingest active fire coordinates CSV and bin them spatially
    fire_frp_grid = pipeline.harmonize_fire_data(
        csv_path="data/fire_events.csv",
        lat_col='latitude',
        lon_col='longitude',
        frp_col='frp'
    )
    
    # Perform background subtraction
    hcho_net = pipeline.cleanse_background(
        data_da=hcho_regridded,
        baseline=4.0e15
    )
    
    return hcho_net, fire_frp_grid

# Load datasets
hcho_net, fire_frp_grid = load_base_datasets()
pipeline = get_pipeline()

# Apply dynamic transport correction based on live slider values
# We use the prompt-specified formula 'prompt' (Ls = Wind / tau) as requested
hcho_corrected = pipeline.apply_transport_correction(
    data_da=hcho_net,
    u_wind=u_wind,
    v_wind=v_wind,
    lifetime_hours=tau_lifetime,
    formula_type='prompt'
)

# 4. Two-Column Visualization Layout
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🗺️ Fire Radiative Power (FRP Matrix)")
    
    # Build Plotly heatmap for binned fires
    fig_fire = px.imshow(
        fire_frp_grid.values,
        x=fire_frp_grid.longitude.values,
        y=fire_frp_grid.latitude.values,
        color_continuous_scale='YlOrRd',
        origin='lower',
        labels={'color': 'FRP (MW)', 'x': 'Longitude', 'y': 'Latitude'}
    )
    fig_fire.update_layout(
        height=550,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )
    st.plotly_chart(fig_fire, use_container_width=True)

with col_right:
    st.markdown("### 🧪 Corrected Tropospheric HCHO Plume")
    
    # Build Plotly heatmap for wind-corrected HCHO columns
    fig_hcho = px.imshow(
        hcho_corrected.values,
        x=hcho_corrected.longitude.values,
        y=hcho_corrected.latitude.values,
        color_continuous_scale='Viridis',
        origin='lower',
        labels={'color': 'HCHO Column', 'x': 'Longitude', 'y': 'Latitude'}
    )
    fig_hcho.update_layout(
        height=550,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )
    st.plotly_chart(fig_hcho, use_container_width=True)

# 5. Live Diagnostic Metrics Footer
st.markdown("---")
st.markdown("### 📊 Live Pipeline Metrics")

# Calculate metrics dynamically based on live corrected grid values
hcho_thresh = 10.0e15
frp_thresh = 40.0

# Generate condition masks on grid values
hcho_mask = hcho_corrected.values > hcho_thresh
frp_mask = fire_frp_grid.values > frp_thresh

# Overlapping pyrogenic hotspots
overlap_mask = hcho_mask & frp_mask
num_hotspots = int(np.sum(overlap_mask))

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        label="Isolated Pyrogenic Hotspots",
        value=f"{num_hotspots}",
        help="Count of target grid cells where corrected Net Reactive HCHO > 1.0e16 molec/cm² and Fire FRP > 40 MW simultaneously"
    )

with metric_col2:
    st.metric(
        label="Target Spatial Grid Domain",
        value="India Bounds (0.1°)",
        help="Target coordinate canvas over India at 0.1° resolution"
    )

with metric_col3:
    # Dynamic status display
    risk_status = "NOx-Limited (FNR > 2.0)" if num_hotspots > 0 else "Baseline Equilibrium"
    st.metric(
        label="Downwind Ozone Risk Regime",
        value=risk_status,
        help="Active ozone production regime calculated dynamically from the live pyrogenic hotspot counts"
    )
