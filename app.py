"""
Interactive Streamlit Dashboard for Multi-Sensor Satellite Data Fusion Pipeline.

This app provides an interactive user interface using Folium to display
binned fire FRP data and wind-aligned chemical plume ratios on Leaflet maps of India.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

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
    
    # Ingest NO2 NetCDF layer (required for FNR ratio bundling)
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
    
    # Load raw fire coordinates for MarkerCluster
    df_raw_fires = pd.read_csv("data/fire_events.csv")
    df_raw_fires_filtered = df_raw_fires[
        (df_raw_fires['latitude'] >= pipeline.lat_min) & (df_raw_fires['latitude'] <= pipeline.lat_max) &
        (df_raw_fires['longitude'] >= pipeline.lon_min) & (df_raw_fires['longitude'] <= pipeline.lon_max)
    ]
    
    return hcho_net, fire_frp_grid, no2_regridded, df_raw_fires_filtered

# Load datasets
hcho_net, fire_frp_grid, no2_regridded, df_raw_fires_filtered = load_base_datasets()
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

# 4. Folium Map Generation Layout
st.markdown("### 🗺️ GIS Atmospheric Monitoring Map (Fires Cluster & Chemical Plumes)")

# Initialize the central Folium Map object focused natively over India
m = folium.Map(
    location=[22.0, 78.0],
    zoom_start=5,
    tiles="CartoDB positron",
    control_scale=True
)

# A. Active Fire Layer: MarkerCluster Plugin
# Groups individual fire markers into numerical clusters dynamically
fire_cluster = MarkerCluster(
    name="🔥 Active Fires Cluster (MODIS/VIIRS)",
    show=True
)

for _, row in df_raw_fires_filtered.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=folium.Icon(color='red', icon='fire', prefix='fa'),
        tooltip=f"Fire intensity: {row['frp']:.1f} MW"
    ).add_to(fire_cluster)

fire_cluster.add_to(m)

# B. Atmospheric Gas Layer (FNR CircleMarkers)
# Dynamic color-coding: values > 2.0 glow bright orange/red for NOx-limited plume
def get_fnr_color(fnr):
    if fnr < 1.0:
        return '#005f73' # VOC-Limited (Dark Blue)
    elif fnr <= 2.0:
        return '#e9d8a6' # Transition (Muted Yellow)
    else:
        return '#ae2012' # NOx-Limited Plume (Bright Red/Orange)

# HTML formatting for popups on element clicks
def create_html_popup(row):
    fnr = row['FNR']
    if fnr < 1.0:
        regime_status = "<span style='color:#005f73; font-weight:bold;'>VOC-Limited Urban</span>"
    elif fnr <= 2.0:
        regime_status = "<span style='color:#ca6702; font-weight:bold;'>Transition Regime</span>"
    else:
        regime_status = "<span style='color:#ae2012; font-weight:bold;'>NOx-Limited Plume</span>"

    html = f"""
    <div style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11px; width: 230px; line-height: 1.6;">
        <h5 style="margin: 0 0 5px 0; color: #2b2d42; font-weight: bold; border-bottom: 2px solid #ddd; padding-bottom: 3px;">Geospatial Canvas Cell</h5>
        <b>Coordinates:</b> [{row['Latitude']:.4f}°N, {row['Longitude']:.4f}°E]<br>
        <b>Fire Intensity (FRP):</b> {row['FRP']:.1f} MW<br>
        <b>Tropospheric HCHO:</b> {row['HCHO']:.3e} molec/cm²<br>
        <b>Tropospheric NO2:</b> {row['NO2']:.3e} molec/cm²<br>
        <b>Chemical Ratio (FNR):</b> {row['FNR']:.2f}<br>
        <b>Regime Status:</b> {regime_status}
    </div>
    """
    return folium.Popup(html, max_width=260)

# Filter grid cells where gas concentrations are active (HCHO > 0)
df_gas = df_aligned[df_aligned['HCHO'] > 0.0]

# Downsample HCHO/FNR CircleMarkers (every 6th pixel) to maintain lag-free rendering
df_gas_downsampled = df_gas.iloc[::6]

gas_markers_group = folium.FeatureGroup(
    name="🧪 Ozone Sensitivity Overlay (FNR Grid)",
    show=True
)

for _, row in df_gas_downsampled.iterrows():
    fnr_val = row['FNR']
    color = get_fnr_color(fnr_val)
    
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        weight=1,
        popup=create_html_popup(row),
        tooltip=f"FNR: {fnr_val:.2f}"
    ).add_to(gas_markers_group)

gas_markers_group.add_to(m)

# C. HCHO Column Density HeatMap Layer (fast canvas-based visualization of the full grid)
hcho_coords = df_gas[['Latitude', 'Longitude', 'HCHO']].dropna()
# Scale values by 1e15 to optimize color weight inside HeatMap Leaflet canvas
heatmap_data = [[row['Latitude'], row['Longitude'], row['HCHO'] / 1.0e15] for _, row in hcho_coords.iterrows()]

hcho_heatmap = HeatMap(
    data=heatmap_data,
    name="☁️ HCHO Density Heatmap",
    min_opacity=0.2,
    radius=15,
    blur=12,
    show=False
)
hcho_heatmap.add_to(m)

# Add layer control so user can toggle layers
folium.LayerControl(position='topright').add_to(m)

# Render map object using st_folium with fixed dimensions and key to avoid refresh center drift
st_folium(m, width=1400, height=600, key="india_geospatial_folium_map")

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
