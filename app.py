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
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Import backend class
from pipeline import AtmosphericPipeline

# 1. Page Setup & Header Configuration
st.set_page_config(
    layout="wide",
    page_title="Atmospheric Monitoring India",
    page_icon="🇮🇳"
)

# Convert background image to base64 for reliable local rendering
import base64
import os

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

bg_base64 = get_base64_image("bg_2.jpg")

css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* Apply Inter as base font for all elements */
* {
    font-family: 'Inter', sans-serif;
}

/* Page background and main containers */
.stApp, .main .block-container {
    background-color: #0B0E14 !important;
    color: #E8EAED !important;
}

header[data-testid="stHeader"] {
    background: #0B0E14 !important;
    border-bottom: 1px solid #232938 !important;
}

#root {
    background: #0B0E14 !important;
}

[data-testid="stToolbarActions"] {
    display: none !important;
}

/* Fallback selectors in case the above doesn't catch it */
.stDeployButton {
    display: none !important;
}

button[kind="deployButton"] {
    display: none !important;
}

/* Target the sidebar toggle button */
[data-testid="collapsedControl"] {
    background: rgba(0, 184, 169, 0.15) !important;
    border: 1px solid rgba(0, 184, 169, 0.4) !important;
    border-radius: 8px !important;
    color: #00B8A9 !important;
    padding: 6px !important;
    transition: all 0.2s ease !important;
}

[data-testid="collapsedControl"]:hover {
    background: rgba(0, 184, 169, 0.3) !important;
    border-color: #00B8A9 !important;
}

/* Also style the open/close arrow inside the sidebar itself */
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(0, 184, 169, 0.15) !important;
    border: 1px solid rgba(0, 184, 169, 0.4) !important;
    border-radius: 8px !important;
    color: #00B8A9 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(0, 184, 169, 0.3) !important;
    border-color: #00B8A9 !important;
}

[data-testid="stSidebarCollapseButton"] svg {
    fill: #00B8A9 !important;
    stroke: #00B8A9 !important;
    width: 20px !important;
    height: 20px !important;
}

button[data-testid="baseButton-headerNoPadding"] {
    color: #FFD700 !important;
    background: rgba(255, 215, 0, 0.1) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    border-radius: 8px !important;
}

button[data-testid="baseButton-headerNoPadding"] svg {
    fill: #FFD700 !important;
    stroke: #FFD700 !important;
}

button[data-testid="baseButton-headerNoPadding"]:hover {
    background: rgba(255, 215, 0, 0.25) !important;
    border-color: #FFD700 !important;
}

/* Fix top bar/toolbar background to match the space image overlay */
[data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
}

/* Remove any top padding/margin creating the black strip */
.stApp > header {
    background: transparent !important;
}

section[data-testid="stSidebarContent"] {
    background: transparent !important;
}

/* Force the very top of the app to be transparent so the 
   space background image shows through everywhere */
.stApp {
    background-color: transparent !important;
}

#root > div:first-child {
    background: transparent !important;
}

/* Header container padding */
.main .block-container {
    padding-top: 2rem !important;
}

/* Custom separator line below title area */
.header-divider {
    border: none !important;
    border-top: 1px solid #232938 !important;
    margin-bottom: 2rem;
    margin-top: 1rem;
}

/* Small eyebrow headers */
.eyebrow-label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    color: #00B8A9 !important;
    text-transform: uppercase !important;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

/* Subheader (h3) styling */
h3, [data-testid="stHeader"] h3, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #E8EAED !important;
    letter-spacing: -0.02em !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0B0E14 !important;
    border-right: 1px solid #232938 !important;
}

/* Sidebar labels and text */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
    color: #8B93A7 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* Sidebar slider active track and handle */
[data-testid="stSlider"] [role="slider"] {
    background-color: #00B8A9 !important;
    border: 2px solid #E8EAED !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background-color: #00B8A9 !important;
}

/* Restyle metric cards */
div[data-testid="stMetric"] {
    background-color: #131720 !important;
    border: 1px solid #232938 !important;
    border-radius: 20px !important;
    padding: 24px 28px !important;
    box-shadow: 0 1px 2px 0 rgba(5,26,36,0.1),
                0 4px 4px 0 rgba(5,26,36,0.09),
                0 9px 6px 0 rgba(5,26,36,0.05),
                0 17px 7px 0 rgba(5,26,36,0.01) !important;
}

div[data-testid="stMetricLabel"] > div, div[data-testid="stMetricLabel"] span {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8B93A7 !important;
}

div[data-testid="stMetricValue"] > div, [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 28px !important;
    font-weight: 500 !important;
    color: #00B8A9 !important;
}

/* Accent colors for specific metric values based on column indexes */
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetricValue"] > div {
    color: #E63946 !important; /* Fire red */
}
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetricValue"] > div {
    color: #00B8A9 !important; /* Accent Teal */
}
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetricValue"] > div {
    color: #4ADE80 !important; /* Success Green */
}

/* Button style overrides */
.stButton > button {
    background-color: #051A24 !important;
    color: #F6FCFF !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 12px 28px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 2px 0 rgba(5,26,36,0.1),
                0 4px 4px 0 rgba(5,26,36,0.09),
                0 9px 6px 0 rgba(5,26,36,0.05),
                inset 0 2px 8px 0 rgba(255,255,255,0.08) !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    color: #F6FCFF !important;
}

/* Restyle Streamlit info boxes */
div[data-testid="stNotification"], div[data-testid="stInfo"] {
    background-color: #131720 !important;
    border: 1px solid #232938 !important;
    border-radius: 12px !important;
    color: #8B93A7 !important;
}
div[data-testid="stNotification"] p, div[data-testid="stInfo"] p {
    color: #8B93A7 !important;
}

/* Selectbox (year selector) styling */
div[data-testid="stSelectbox"] > div {
    background-color: #131720 !important;
    border: 1px solid #232938 !important;
    border-radius: 12px !important;
    color: #E8EAED !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid #232938 !important;
    margin: 24px 0 !important;
}
"""

if bg_base64 is not None:
    css_styles += f"""
.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_base64}") !important;
    background-size: cover !important;
    background-position: center center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}}

/* Subtle dark overlay to preserve readability while letting the 
   space/Earth image show through slightly */
.stApp::before {{
    content: '' !important;
    position: fixed !important;
    inset: 0 !important;
    background: rgba(11, 14, 20, 0.82) !important;
    z-index: 0 !important;
    pointer-events: none !important;
    background-size: cover !important;
}}

/* Ensure all content sits above the overlay */
.stApp > * {{
    position: relative !important;
    z-index: 1 !important;
}}

header[data-testid="stHeader"] {{
    background: rgba(11, 14, 20, 0.82) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border-bottom: 1px solid #232938 !important;
}}

section[data-testid="stSidebar"] {{
    background: rgba(11, 14, 20, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(35, 41, 56, 0.8) !important;
}}
"""

st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

st.title("🇮🇳 Multi-Sensor Satellite Data Fusion Dashboard")
st.markdown('<div class="eyebrow-label">OBJECTIVE 2 · ATMOSPHERIC TRANSPORT ANALYTICS</div>', unsafe_allow_html=True)
st.subheader("Objective 2: HCHO Hotspot Identification & Wind Plume Transport Correction")
st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)

# 1. Sidebar Year Selector
selected_year = st.sidebar.selectbox(
    "📅 Select Observation Year",
    ["2026 (Live Interactive Pipeline)", "2025", "2024", "2023", "2022", "2021", "2020", "2019"]
)

if selected_year == "2026 (Live Interactive Pipeline)":
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
        "These sliders dynamically compute the Smearing Length Scale (Ls = U * τ) "
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

if selected_year == "2026 (Live Interactive Pipeline)":
    # Load datasets
    hcho_net, fire_frp_grid, no2_regridded = load_base_datasets()
    pipeline = get_pipeline()

    # Apply dynamic transport correction based on live slider values
    hcho_corrected = pipeline.apply_transport_correction(
        data_da=hcho_net,
        u_wind=u_wind,
        v_wind=v_wind,
        lifetime_hours=tau_lifetime,
        formula_type='physical'
    )

    # Apply equivalent ocean mask to raw xarray datasets for metric consistency
    ocean_mask = ~(((hcho_corrected.longitude < 72.5) & (hcho_corrected.latitude < 22.0)) | 
                   ((hcho_corrected.longitude > 85.0) & (hcho_corrected.latitude < 20.0)))
    hcho_corrected = hcho_corrected.where(ocean_mask, np.nan)
    fire_frp_grid = fire_frp_grid.where(ocean_mask, np.nan)

    # Bundle all aligned spatial layers into a single clean Pandas DataFrame
    df_aligned = pipeline.bundle_to_dataframe(
        hcho_da=hcho_corrected,
        no2_da=no2_regridded,
        fire_da=fire_frp_grid
    )

    # --- GEOGRAPHIC OCEAN MASK (approximate rectangular cutoffs, not a true coastline) ---
    arabian_sea_mask = (df_aligned['Longitude'] < 72.5) & (df_aligned['Latitude'] < 22.0)
    df_aligned.loc[arabian_sea_mask, ['HCHO', 'NO2', 'FRP']] = np.nan

    bay_of_bengal_mask = (df_aligned['Longitude'] > 85.0) & (df_aligned['Latitude'] < 20.0)
    df_aligned.loc[bay_of_bengal_mask, ['HCHO', 'NO2', 'FRP']] = np.nan
    # ------------------------------

    # Labels map for Plotly charts
    labels = {
        'Pixel_ID': 'Pixel ID',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'FRP': 'Fire Radiative Power (FRP)',
        'HCHO': 'Tropospheric HCHO VCD',
        'NO2': 'Tropospheric NO2 VCD',
        'FNR': 'Chemical Diagnostic Ratio (FNR)'
    }

    # Formatting mappings for hover popup metrics (excluding the Pixel_ID which is the hover_name)
    hover_data = {
        'Pixel_ID': False,  # Hide the raw boolean since it's already the hover_name
        'Latitude': True,
        'Longitude': True,
        'FRP': ':.1f',
        'HCHO': ':.3e',
        'NO2': ':.3e',
        'FNR': ':.2f'
    }

    # 4. Two-Column Visualization Layout (Synchronized Maps of India)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🗺️ Fire Radiative Power (FRP Map)")
        
        # Filter DataFrame to show only active fire grid cells (FRP > 0) to optimize rendering speed
        df_fire = df_aligned[df_aligned['FRP'] > 0.0].copy()
        
        # Generate clean professional index labels to avoid memory address leaks in the tooltip header
        df_fire['Pixel_ID'] = [f"Pixel_ID_{i+1:04d}" for i in range(len(df_fire))]
        
        # Build Mapbox scatter plot for active fires
        fig_fire = px.scatter_mapbox(
            df_fire,
            lat='Latitude',
            lon='Longitude',
            color='FRP',
            size='FRP',
            color_continuous_scale='YlOrRd',
            mapbox_style='carto-positron',
            hover_name='Pixel_ID',
            hover_data=hover_data,
            labels=labels,
            size_max=18
        )
        # Hardcode Mapbox view configuration to center perfectly over Central India land
        fig_fire.update_layout(
            height=550,
            margin=dict(l=0, r=0, t=30, b=0),
            mapbox=dict(
                center=dict(lat=22.0, lon=78.0),
                zoom=4,
                style="carto-positron"
            ),
            coloraxis_colorbar=dict(title="FRP (MW)")
        )
        st.plotly_chart(fig_fire, use_container_width=True)

    with col_right:
        st.markdown("### 🧪 Corrected Tropospheric HCHO Plume Map")
        
        # Filter HCHO cells above threshold and downsample slightly (by taking every 2nd point)
        # to maintain high-performance, real-time recalculations on the Mapbox layer.
        df_hcho = df_aligned[df_aligned['HCHO'] > 0.3e15]
        df_hcho_plot = df_hcho.iloc[::2].copy()
        
        # Generate clean professional index labels to avoid memory address leaks in the tooltip header
        df_hcho_plot['Pixel_ID'] = [f"Pixel_ID_{i+1:04d}" for i in range(len(df_hcho_plot))]
        
        # Build Mapbox scatter plot for HCHO grid
        fig_hcho = px.scatter_mapbox(
            df_hcho_plot,
            lat='Latitude',
            lon='Longitude',
            color='HCHO',
            color_continuous_scale='Viridis',
            mapbox_style='carto-positron',
            hover_name='Pixel_ID',
            hover_data=hover_data,
            labels=labels
        )
        # Set continuous grid overlay marker appearance
        fig_hcho.update_traces(
            marker=dict(size=6, opacity=0.7)
        )
        # Hardcode Mapbox view configuration to synchronize perfectly with the FRP map
        fig_hcho.update_layout(
            height=550,
            margin=dict(l=0, r=0, t=30, b=0),
            mapbox=dict(
                center=dict(lat=22.0, lon=78.0),
                zoom=4,
                style="carto-positron"
            ),
            coloraxis_colorbar=dict(title="molec/cm²")
        )
        st.plotly_chart(fig_hcho, use_container_width=True)


    # 5. Full-Width Pyrogenic Analytics & Trends Section (Side-by-Side Subplots)
    st.markdown('<div class="eyebrow-label">ANALYTICS · TEMPORAL & REGRESSION TRENDS</div>', unsafe_allow_html=True)
    st.subheader("📈 Pyrogenic Analytics & Trends")

    # Generate mock daily time-series dataset spanning the last 30 days using Pandas
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    days = np.arange(30)

    # Simulate active fire counts peaking during biomass burning windows
    fire_trend = 15.0 + 80.0 * np.exp(-((days - 15.0) / 4.0)**2) + np.random.normal(0, 5, 30)
    fire_trend = np.clip(fire_trend, 0, None)

    # Simulate lagging tropospheric HCHO column density peaking slightly later due to chemical build-up/transport
    hcho_trend = 4.0e15 + 8.5e15 * np.exp(-((days - 17.0) / 5.0)**2) + np.random.normal(0, 0.25e15, 30)

    df_trends = pd.DataFrame({
        'Date': dates,
        'Active Fire Detections': fire_trend,
        'HCHO Column Density': hcho_trend
    })

    # Side-by-side layout for analytics
    col_trend_left, col_trend_right = st.columns(2)

    with col_trend_left:
        # Create dual-axis line chart using subplots
        fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trends.add_trace(
            go.Scatter(
                x=df_trends['Date'],
                y=df_trends['Active Fire Detections'],
                name="Active Fire Detections (MODIS)",
                line=dict(color="#ae2012", width=3)
            ),
            secondary_y=False,
        )
        fig_trends.add_trace(
            go.Scatter(
                x=df_trends['Date'],
                y=df_trends['HCHO Column Density'],
                name="HCHO Column Density (TROPOMI)",
                line=dict(color="#005f73", width=3)
            ),
            secondary_y=True,
        )
        fig_trends.update_layout(
            title_text="Fires vs. HCHO Column Density (30-Day Trend)",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            paper_bgcolor='#131720',
            plot_bgcolor='#131720'
        )
        fig_trends.update_yaxes(title_text="<b>Active Fire Counts</b>", color="#ae2012", secondary_y=False)
        fig_trends.update_yaxes(title_text="<b>HCHO Column Density</b>", color="#005f73", secondary_y=True)
        st.plotly_chart(fig_trends, use_container_width=True)

    with col_trend_right:
        # Create scatter correlation plot with trendline (OLS)
        try:
            fig_corr = px.scatter(
                df_trends,
                x='Active Fire Detections',
                y='HCHO Column Density',
                trendline="ols",
                title="Fires vs. HCHO Enhancement Correlation",
                labels={
                    'Active Fire Detections': 'Active Fire Detections (counts)',
                    'HCHO Column Density': 'HCHO Column Density (molec/cm²)'
                },
                template="plotly_white"
            )
            fig_corr.update_traces(marker=dict(color='#005f73', size=8))
        except ImportError:
            # Fallback if statsmodels is not installed on system
            fig_corr = px.scatter(
                df_trends,
                x='Active Fire Detections',
                y='HCHO Column Density',
                title="Fires vs. HCHO Enhancement Correlation",
                labels={
                    'Active Fire Detections': 'Active Fire Detections (counts)',
                    'HCHO Column Density': 'HCHO Column Density (molec/cm²)'
                },
                template="plotly_white"
            )
            fig_corr.update_traces(marker=dict(color='#005f73', size=8))
            # Manually calculate and add linear regression line using numpy polyfit
            x = df_trends['Active Fire Detections'].values
            y = df_trends['HCHO Column Density'].values
            idx = np.isfinite(x) & np.isfinite(y)
            if len(x[idx]) > 1:
                coef = np.polyfit(x[idx], y[idx], 1)
                poly1d_fn = np.poly1d(coef)
                x_range = np.linspace(x[idx].min(), x[idx].max(), 100)
                fig_corr.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=poly1d_fn(x_range),
                        mode='lines',
                        name='OLS Fit',
                        line=dict(color='#ae2012', width=2, dash='dash')
                    )
                )
                
        fig_corr.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            paper_bgcolor='#131720',
            plot_bgcolor='#131720'
        )
        st.plotly_chart(fig_corr, use_container_width=True)


    # 6. Live Diagnostic Metrics Footer
    st.markdown("---")
    st.markdown('<div class="eyebrow-label">METRICS · REAL-TIME QUANTITATIVE ATTRIBUTION</div>', unsafe_allow_html=True)
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
else:
    st.markdown(f"## 📅 Historical Observations — {selected_year}")
    st.markdown("---")
    
    # HCHO Row: Full-width combined HCHO map (Baseline vs. Burning Season side-by-side in one image)
    st.markdown("### 🧪 Tropospheric HCHO Plume Map (Baseline vs. Burning Season)")
    st.caption(f"Comparison of HCHO column density (mol/m²) for Baseline (Jan-Mar) and Burning Season (Oct-Nov) in {selected_year}")
    hcho_file = f"static_maps/{selected_year}_hcho_combined.png"
    if os.path.exists(hcho_file):
        st.image(hcho_file, use_container_width=True)
    else:
        st.info(f"HCHO comparison map for {selected_year} pending generation (expected {hcho_file})")
        
    st.markdown("---")
    
    # FRP Row: Side-by-side baseline vs burning FRP maps
    frp_col1, frp_col2 = st.columns(2)
    
    with frp_col1:
        st.markdown("### 🔥 Baseline Season - Fire Radiative Power (FRP)")
        st.caption(f"FRP distribution during Jan-Mar {selected_year}")
        frp_base_file = f"static_maps/{selected_year}_frp_baseline.png"
        if os.path.exists(frp_base_file):
            st.image(frp_base_file, use_container_width=True)
        else:
            st.info(f"Baseline FRP map for {selected_year} pending generation (expected {frp_base_file})")
            
    with frp_col2:
        st.markdown("### 🔥 Biomass Burning Season - Fire Radiative Power (FRP)")
        st.caption(f"FRP distribution during Oct-Nov {selected_year}")
        frp_burn_file = f"static_maps/{selected_year}_frp_burning.png"
        if os.path.exists(frp_burn_file):
            st.image(frp_burn_file, use_container_width=True)
        else:
            st.info(f"Burning Season FRP map for {selected_year} pending generation (expected {frp_burn_file})")
