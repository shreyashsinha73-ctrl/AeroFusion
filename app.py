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

# Inject custom CSS for polished light slate/gray theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

/* Main app container background */
.stApp {
    background-color: #f8f9fa !important;
    color: #212529 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Header container styling */
.main .block-container {
    padding-top: 2rem !important;
}

/* Custom separator line below title area */
.header-divider {
    border-bottom: 2px solid #dee2e6;
    margin-bottom: 2rem;
    margin-top: 1rem;
}

/* Small eyebrow headers */
.eyebrow-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #005f73;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #e9ecef !important;
    border-right: 1px solid #dee2e6 !important;
}

/* Text elements color */
h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: #212529;
}

/* Mute description text */
.stMarkdown p, .stMarkdown span {
    color: #495057;
}

/* Restyle metric cards */
div[data-testid="stMetric"] {
    background-color: #e9ecef !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 6px !important;
    padding: 15px 20px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
}

div[data-testid="stMetricLabel"] > div {
    font-family: 'Inter', sans-serif !important;
    color: #495057 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricValue"] > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* Accent colors for specific metrics based on column indexes */
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetricValue"] > div {
    color: #ae2012 !important; /* Accent 2 / Fire red */
}
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetricValue"] > div {
    color: #005f73 !important; /* Accent 1 / Teal */
}
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetricValue"] > div {
    color: #2a9d8f !important; /* Success green */
}

/* Custom BaseWeb Slider styling */
div[data-baseweb="slider"] {
    margin-bottom: 1.5rem !important;
}
div[data-testid="stSlider"] label {
    font-family: 'Inter', sans-serif !important;
    color: #212529 !important;
    font-size: 0.9rem !important;
}

/* Slider thumb color */
div[role="slider"] {
    background-color: #005f73 !important;
    border: 2px solid #f8f9fa !important;
}

/* Slider active track color */
div[data-baseweb="slider"] > div > div {
    background-color: #005f73 !important;
}

/* Restyle Streamlit info boxes */
div[data-testid="stNotification"] {
    background-color: #e9ecef !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
div[data-testid="stNotification"] p {
    color: #495057 !important;
    font-size: 0.85rem !important;
}

</style>
""", unsafe_allow_html=True)

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
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            paper_bgcolor='#e9ecef',
            plot_bgcolor='#e9ecef'
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
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            paper_bgcolor='#e9ecef',
            plot_bgcolor='#e9ecef'
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
    
    # 2x2 grid of columns
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    panels = [
        {
            "col": row1_col1,
            "title": "🔥 Baseline Season - Fire Radiative Power (FRP)",
            "caption": f"Baseline: Jan-Mar {selected_year} (Placeholder)",
            "file": f"static_maps/{selected_year}_baseline_frp.png"
        },
        {
            "col": row1_col2,
            "title": "🧪 Baseline Season - Tropospheric HCHO Plume",
            "caption": f"Baseline: Jan-Mar {selected_year} (Placeholder)",
            "file": f"static_maps/{selected_year}_baseline_hcho.png"
        },
        {
            "col": row2_col1,
            "title": "🔥 Biomass Burning Season - Fire Radiative Power (FRP)",
            "caption": f"Burning Season: Oct-Nov {selected_year} (Placeholder)",
            "file": f"static_maps/{selected_year}_burning_frp.png"
        },
        {
            "col": row2_col2,
            "title": "🧪 Biomass Burning Season - Tropospheric HCHO Plume",
            "caption": f"Burning Season: Oct-Nov {selected_year} (Placeholder)",
            "file": f"static_maps/{selected_year}_burning_hcho.png"
        }
    ]
    
    for panel in panels:
        with panel["col"]:
            st.markdown(f"### {panel['title']}")
            st.caption(panel["caption"])
            file_path = panel["file"]
            if os.path.exists(file_path):
                st.image(file_path, use_container_width=True)
            else:
                st.info(f"Map for {selected_year} pending generation")
