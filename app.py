"""
Interactive Streamlit Dashboard for Multi-Sensor Satellite Data Fusion Pipeline.

This app provides an interactive user interface using Plotly Express Mapbox maps to display
binned fire FRP data and wind-aligned chemical plume columns over India.
"""

import os
import numpy as np
import pandas as pd
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

# Sidebar view domain selector
view_domain = st.sidebar.selectbox(
    label="🔍 Select View Domain",
    options=["National Overview (All India)", "Indo-Gangetic Plain (Hotspot Close-up)"],
    help="Select the geographical focus and zoom level for the Mapbox charts."
)

# Set map zoom and center dynamically based on selectbox option
if view_domain == "National Overview (All India)":
    map_center = dict(lat=22.0, lon=78.0)
    map_zoom = 4
else:
    map_center = dict(lat=22.0, lon=80.0)
    map_zoom = 7

st.sidebar.info(
    "These sliders dynamically compute the Smearing Length Scale (Ls ≈ U / τ) "
    "in real-time, inversely shifting the TROPOMI HCHO column matrices back "
    "to the true MODIS/VIIRS fire coordinates."
)

# 3. Backend Integration & Data Loading
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
    
    # Ingest NO2 NetCDF layer
    no2_regridded = pipeline.ingest_satellite_data(
        file_path="data/no2_raw.nc",
        variable_name='no2_column',
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
    
    return hcho_net, fire_frp_grid, no2_regridded

# Load datasets
hcho_net, fire_frp_grid, no2_regridded = load_base_datasets()
pipeline = get_pipeline()

# Apply dynamic transport correction based on live slider values
hcho_corrected = pipeline.apply_transport_correction(
    data_da=hcho_net,
    u_wind=u_wind,
    v_wind=v_wind,
    lifetime_hours=tau_lifetime,
    formula_type='prompt'
)

# Bundle all aligned spatial layers into a single clean Pandas DataFrame
df_aligned = pipeline.bundle_to_dataframe(
    hcho_da=hcho_corrected,
    no2_da=no2_regridded,
    fire_da=fire_frp_grid
)

# Labels map for Plotly charts
labels = {
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'FRP': 'Fire Radiative Power (FRP)',
    'HCHO': 'Tropospheric HCHO VCD',
    'NO2': 'Tropospheric NO2 VCD',
    'FNR': 'Chemical Diagnostic Ratio (FNR)'
}

# Formatting mappings for hover popup metrics
hover_data = {
    'Latitude': True,
    'Longitude': True,
    'FRP': ':.1f',
    'HCHO': ':.3e',
    'NO2': ':.3e',
    'FNR': ':.2f'
}

# 4. Two-Column Visualization Layout
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🗺️ Fire Radiative Power (FRP Map)")
    
    # Filter DataFrame to show only active fire grid cells (FRP > 0) to optimize rendering speed
    df_fire = df_aligned[df_aligned['FRP'] > 0.0]
    
    # Build Mapbox scatter plot for active fires
    fig_fire = px.scatter_mapbox(
        df_fire,
        lat='Latitude',
        lon='Longitude',
        color='FRP',
        size='FRP',
        color_continuous_scale='YlOrRd',
        zoom=map_zoom,
        center=map_center,
        mapbox_style='carto-positron',
        hover_name='FRP',
        hover_data=hover_data,
        labels=labels,
        size_max=18
    )
    fig_fire.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="FRP (MW)")
    )
    st.plotly_chart(fig_fire, use_container_width=True)

with col_right:
    st.markdown("### 🧪 Corrected Tropospheric HCHO Plume Map")
    
    # Filter HCHO cells > 0 and downsample slightly (by taking every 2nd point)
    # to maintain high-performance, real-time recalculations on the Mapbox layer.
    df_hcho = df_aligned[df_aligned['HCHO'] > 0.0]
    df_hcho_plot = df_hcho.iloc[::2]
    
    # Build Mapbox scatter plot for HCHO grid
    fig_hcho = px.scatter_mapbox(
        df_hcho_plot,
        lat='Latitude',
        lon='Longitude',
        color='HCHO',
        color_continuous_scale='Viridis',
        zoom=map_zoom,
        center=map_center,
        mapbox_style='carto-positron',
        hover_name='HCHO',
        hover_data=hover_data,
        labels=labels
    )
    # Set continuous grid overlay marker appearance
    fig_hcho.update_traces(
        marker=dict(size=6, opacity=0.7)
    )
    fig_hcho.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="molec/cm²")
    )
    st.plotly_chart(fig_hcho, use_container_width=True)

# 5. Live Diagnostic Metrics Footer
st.markdown("---")
st.markdown("### 📊 Live Pipeline Metrics")

# Calculate metrics dynamically using 2D arrays for optimization
hcho_thresh = 10.0e15
frp_thresh = 40.0

hcho_mask = hcho_corrected.values > hcho_thresh
frp_mask = fire_frp_grid.values > frp_thresh

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
    risk_status = "NOx-Limited (FNR > 2.0)" if num_hotspots > 0 else "Baseline Equilibrium"
    st.metric(
        label="Downwind Ozone Risk Regime",
        value=risk_status,
        help="Active ozone production regime calculated dynamically from the live pyrogenic hotspot counts"
    )
