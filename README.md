# 🇮🇳 Multi-Sensor Satellite Data Fusion for Atmospheric Monitoring over India

An end-to-end data processing, regridding, and chemical attribution pipeline linking raw satellite sensor observations directly to downwind surface ozone air quality risks.

---

## 1. Architecture Overview

This project implements a multi-sensor satellite data fusion engine that ingests, cleans, aligns, and processes atmospheric products (HCHO, $\text{NO}_2$, and Active Fires) over the Indian subcontinent. The architecture is split into two core modules:

```
+-------------------------------------------------------------------------------+
|                      OBJECTIVE 1: Geospatial Grid Canvas                      |
|                                                                               |
|   [TROPOMI HCHO/NO2 NetCDF]       [MODIS/VIIRS Active Fires CSV]              |
|               |                                  |                            |
|        (QA/Cloud Filter)               (Bounding Box Filter)                  |
|        [qa >= 0.5, cc <= 0.4]          [8.0°N-37.5°N, 68.0°E-97.5°E]          |
|               |                                  |                            |
|       (1D/2D Regridding)               (2D Spatial Binning)                   |
|               +-----------------+----------------+                            |
|                                 |                                             |
|                                 v                                             |
|                    [Aligned 0.1° Target Grid Canvas]                          |
+-------------------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------------------+
|                   OBJECTIVE 2: Atmospheric Transport Analytics                |
|                                                                               |
|                   [HCHO columns]               [Fire FRP Grid]                |
|                          |                            |                       |
|               (Background Subtraction)                |                       |
|                          |                            |                       |
|              (Wind Transport Correction)              |                       |
|               [Ls = Wind / tau shift]                 |                       |
|                          +--------------+-------------+                       |
|                                         |                                     |
|                                         v                                     |
|                        [Spatial Cross-Correlation Engine]                     |
|                                         |                                     |
|                                         v                                     |
|                        [FNR Ozone Sensitivity Profiler]                       |
+-------------------------------------------------------------------------------+
```

### Module A: Geospatial Grid Infrastructure (Objective 1)
- **Bounding Box Slicing**: Limits geographic boundaries strictly to Latitude $8.0^\circ\text{N}$ to $37.5^\circ\text{N}$ and Longitude $68.0^\circ\text{E}$ to $97.5^\circ\text{E}$.
- **Quality Assurance Array Filtering**: Filters out pixels where $QA\_value < 0.5$ or $cloud\_fraction > 0.4$ to clear out cloud and haze contaminations.
- **Point-to-Pixel Standardization**: Re-projects raw polar swaths (unstructured 2D coordinate fields) and regular rectilinear satellite layers onto a uniform $0.1^\circ$ resolution target grid canvas. Active MODIS/VIIRS fire coordinates are harmonized using weighted 2D spatial binning (`numpy.histogram2d`) with Fire Radiative Power (FRP) serving as weights.

### Module B: Atmospheric Transport Analytics (Objective 2)
- **Seasonal Background Cleansing**: Subtracts clean-season baseline values to isolate pure pyrogenic vertical column concentrations.
- **Wind Plume Realignment**: Calculates the transport-displacement scale using 850 hPa wind fields and VOC lifetime to trace shifted downwind plumes back to their fire hotspots.
- **Ozone Risk Profiling**: Uses the Formaldehyde-to-$\text{NO}_2$ ratio ($FNR$) to classify chemical production sensitivity and evaluate downstream pollution surges.

---

## 2. Repository Structure

```
isro-satellite-fusion/
├── pipeline.py        # Vectorized NumPy & Xarray data broker backend engine
├── app.py             # Interactive Streamlit frontend visualization dashboard
└── README.md          # Project system implementation documentation (this file)
```

---

## 3. Installation & Deployment Guide

### Step 1: Clone the Repository
Clone this repository to your local system and navigate to the directory:
```bash
git clone https://github.com/your-username/isro-satellite-fusion.git
cd isro-satellite-fusion
```

### Step 2: Install Dependencies
Install all required libraries into your Python environment:
```bash
pip install numpy pandas xarray scipy streamlit plotly netcdf4
```

### Step 3: Run the Application

#### A. Run the Interactive Dashboard
To launch the interactive web interface, run:
```bash
streamlit run app.py
```
Open the printed local URL (usually `http://localhost:8501`) in your browser to interact with the wind transport sliders and metrics dashboard.

#### B. Run the CLI Pipeline (Backend Verification)
To execute the pipeline directly in the console and export results to a NetCDF file, run:
```bash
python pipeline.py --demo
```
This automatically generates a synthetic dataset in the `data/` directory, runs the full processing engine, and saves output matrices to `output_fusion_results.nc`.

---

## 4. Core Theoretical Framework

The backend engine implements the following core models:

### A. Baseline Cleansing & Methane Subtracting
To isolate chemical enhancements from pyrogenic activities, the seasonal background column is subtracted from observed vertical columns:

$$\Omega_{\text{reactive}} = \max\left(0, \Omega_{\text{observed}} - \Omega_{\text{baseline}}\right)$$

Where:
- $\Omega_{\text{observed}}$ is the regridded satellite observation column ($\text{molec/cm}^2$).
- $\Omega_{\text{baseline}}$ is the clean-season reference baseline constant set at $4.0 \times 10^{15} \text{ molec/cm}^2$.
- Negative column remainders are clipped to $0.0$ to ensure physical consistency.

### B. Wind Plume Realignment
Secondary atmospheric products (such as HCHO) are translated downwind during their chemical formation lifetime. The physical translation offset ($\vec{L}_s$) is computed using wind velocities at the boundary layer pressure height (850 hPa) and the VOC chemical lifetime ($\tau$):

$$\vec{L}_s = \frac{\vec{u}}{\tau}$$

Where:
- $\vec{u} = (U, V)$ is the wind velocity components ($m/s$).
- $\tau$ is the VOC precursor lifetime ($s$).
- The coordinate translation is inverted ($\vec{x}_{\text{lookup}} = \vec{x}_{\text{grid}} + \vec{L}_s$) and mapped using sub-pixel bilinear interpolation (`scipy.ndimage.map_coordinates`) to trace HCHO columns back to their true pyrogenic fire sources:

$$\Delta y_{\text{deg}} = \frac{V_s}{R_{\text{lat}}}$$

$$\Delta x_{\text{deg}} = \frac{U_s}{R_{\text{lon}} \cdot \cos\left(\theta\right)}$$

Where $R_{\text{lat}} = R_{\text{lon}} \approx 111,120 \text{ meters/degree}$ and $\theta$ is latitude in radians.

### C. Ozone Risk Regimes
The Formaldehyde-to-$\text{NO}_2$ Ratio ($FNR$) acts as a proxy for tracking downwind secondary pollution surges ($5$ to $20\text{ ppb}$ in toxic surface ozone):

$$\text{Regime} = \begin{cases} 
\text{VOC-Limited} & \text{if } FNR < 1.0 \\ 
\text{Transition/Mixed} & \text{if } 1.0 \le FNR \le 2.0 \\ 
\text{NOx-Limited} & \text{if } FNR > 2.0 
\end{cases}$$

- **VOC-Limited**: Characteristic of dense urban chemistry footprints (high local $\text{NO}_2$ from combustion).
- **NOx-Limited**: Characteristic of pyrogenic biomass plume footprints (intense VOC/HCHO emissions from wood burning).
- **Transition/Mixed**: High chemical sensitivity transition zone.
