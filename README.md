# 🇮🇳 Multi-Sensor Satellite Data Fusion & Deep Learning for Air Quality Mapping over India

An advanced end-to-end data processing and predictive machine learning pipeline linking multi-sensor remote observations directly to surface-level Air Quality Index (AQI) risks and chemical attribution dynamics.

---

## 1. Project Objectives

The project coordinates remote sensing observations with ground-truth networks to resolve two primary objectives:

```
+---------------------------------------------------------------------------------------------------+
|                           METRIC INGESTION & DATA STORAGE PIPELINE (pipeline.py)                  |
|                                                                                                   |
|   [TROPOMI Satellite]       [CPCB Ground Stations]       [Reanalysis Weather]       [MODIS/VIIRS] |
|     (HCHO & NO2 VCDs)        (PM2.5, PM10, SO2, O3)      (IMDAA, ERA5, MERRA2)      (Active Fires)|
+---------------------------------------------------------------------------------------------------+
                               |                                   |
                               v                                   v
+-------------------------------------------------------------+ +-------------------------------------+
|         OBJECTIVE 1: Deep Learning Surface AQI              | |  OBJECTIVE 2: Plume Transport Shift |
|                                                             | |                                     |
|    +---------------------------------------------------+    | |    [Background Cleansing Subtraction]  |
|    |      Spatial 2D Feature Extraction (CNN Layer)    |    | |                    |                |
|    +---------------------------------------------------+    | |     [Wind Coordinate Roll Shift]    |
|                              |                              | |       (Ls = Wind / tau formula)     |
|                              v                              | |                    |                |
|    +---------------------------------------------------+    | |      [FNR Regime Classification]    |
|    |      Temporal Sequence Prediction (LSTM Layer)    |    | |         (VOC- vs. NOx-Limited)      |
|    +---------------------------------------------------+    | +-------------------------------------+
|                              |                                                   |
|                              v                                                   v
+---------------------------------------------------------------------------------------------------+
|                        STREAMLIT GEOSPATIAL DASHBOARD INTERFACE (app.py)                          |
+---------------------------------------------------------------------------------------------------+
```

### Objective 1: Surface AQI Estimation via Spatial Deep Learning
- **Multi-source Data Ingestion**: Integrates columnar satellite columns (TROPOMI HCHO, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$), surface-level ground monitors (Central Pollution Control Board - CPCB network), and multi-source meteorological reanalysis grids (IMDAA, ERA5, and MERRA-2).
- **Hybrid CNN-LSTM Architecture**: Combines a Convolutional Neural Network (CNN) to extract spatial dependencies from the 2D gridded matrices and a Long Short-Term Memory (LSTM) recurrent network to capture temporal accumulation and dispersion sequences. This allows predicting continuous ground-level pollutant maps and calculating true surface AQI over the Indian subcontinent.

### Objective 2: Pyrogenic HCHO Hotspot & Atmospheric Transport Analytics
- **Active Fire Harmonization**: Uses MODIS and VIIRS thermal coordinates to map fire outbreaks and extract fire count arrays during agricultural burning windows.
- **Spatio-Temporal Plume Tracking**: Ingests TROPOMI Formaldehyde (HCHO) to isolate major pyrogenic emissions over fire-impacted zones like the Indo-Gangetic Plain.
- **Wind Transport Shift Correction**: Dynamically adjusts displaced downwind HCHO columns back to their true pyrogenic fire sources using boundary layer wind vectors.

---

## 2. Core Theoretical Framework

The processing and deep learning modules implement the following mathematical frameworks:

### A. Hybrid Spatial-Temporal Modeling (Objective 1)
To capture spatial layouts and temporal trends, we stack 2D spatial convolutions with recurrent LSTM nodes:

#### Spatial Feature Extraction (CNN)
The 2D spatial convolutional layer extracts regional meteorological and satellite features:

$$H_{i,j,k} = \sigma \left( \sum_{m=-M}^{M} \sum_{n=-N}^{N} \sum_{c=1}^{C} W_{m,n,c,k} \cdot X_{i+m, j+n, c} + b_k \right)$$

Where:
- $X$ is the input geospatial tensor containing satellite columns, meteorological layers, and CPCB interpolation grids.
- $W_{m,n,c,k}$ is the weights kernel of the $k$-th filter.
- $b_k$ is the bias vector, and $\sigma$ represents the activation function (e.g., Rectified Linear Unit - ReLU).

#### Temporal sequence tracking (LSTM)
The spatial features are flattened and fed to an LSTM network to model pollutant accumulation timelines:

$$f_t = \sigma_g \left( W_f \cdot [h_{t-1}, x_t] + b_f \right)$$

$$i_t = \sigma_g \left( W_i \cdot [h_{t-1}, x_t] + b_i \right)$$

$$\tilde{C}_t = \tanh \left( W_c \cdot [h_{t-1}, x_t] + b_c \right)$$

$$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$

$$o_t = \sigma_g \left( W_o \cdot [h_{t-1}, x_t] + b_o \right)$$

$$h_t = o_t \cdot \tanh(C_t)$$

Where:
- $x_t$ is the spatial feature vector extracted by the CNN at timestep $t$.
- $f_t, i_t, o_t$ are the forget, input, and output gate activation vectors.
- $C_t$ is the internal cell state, and $h_t$ is the hidden state vector output representing the predicted surface AQI values.

---

### B. Meteorological Wind Transport Shift (Objective 2)
Volatile Organic Compounds (VOCs) undergo wind transport during their chemical oxidation lifetime ($\tau$). To trace Observed HCHO columns back to their pyrogenic sources, we calculate the Smearing Length Scale ($\vec{L}_s$):

$$\vec{L}_s = \frac{\vec{u}}{\tau}$$

Where:
- $\vec{u} = (U, V)$ is the boundary layer wind velocity vectors ($m/s$) at 850 hPa.
- $\tau$ is the VOC precursor chemical lifetime ($s$) before decaying into HCHO.
- The inverse coordinate roll shift is computed using grid cell distances:

$$\Delta y_{\text{deg}} = \frac{V_s}{R_{\text{lat}}}$$

$$\Delta x_{\text{deg}} = \frac{U_s}{R_{\text{lon}} \cdot \cos\left(\theta\right)}$$

Where $R_{\text{lat}} = R_{\text{lon}} \approx 111,120 \text{ meters/degree}$ and $\theta$ is the latitude in radians.

---

### C. Chemical Footprint Attribution (Objective 2)
We evaluate downstream ozone production regimes using the Formaldehyde-to-$\text{NO}_2$ Ratio ($FNR$):

$$\text{Regime} = \begin{cases} 
\text{VOC-Limited} & \text{if } FNR < 1.0 \\ 
\text{Transition/Mixed} & \text{if } 1.0 \le FNR \le 2.0 \\ 
\text{NOx-Limited} & \text{if } FNR > 2.0 
\end{cases}$$

Where $FNR = \frac{[\text{HCHO}]}{[\text{NO}_2]}$ represents the ratio of the vertical column densities. This diagnostic evaluates secondary pollution surges ($5$ to $20 \text{ ppb}$ in toxic surface ozone).

---

## 3. Repository Directory Structure

```
isro-satellite-fusion/
├── pipeline.py             # xarray/NumPy vectorized data broker and binning engine
├── app.py                  # Interactive Streamlit frontend simulation dashboard
├── models/
│   ├── cnn_lstm_trainer.py # CNN-LSTM training pipeline script
│   └── checkpoints/        # Model weight checkpoint files (.h5 / .pt)
└── README.md               # System documentation (this file)
```

---

## 4. System Installation & Quickstart

### Step 1: Clone the Repository
Clone the repository and enter the project folder:
```bash
git clone https://github.com/your-username/isro-satellite-fusion.git
cd isro-satellite-fusion
```

### Step 2: Install Dependencies
Install all required libraries (including deep learning and geospatial libraries):
```bash
pip install numpy pandas xarray scipy tensorflow torch streamlit plotly netcdf4
```

### Step 3: Launch the Interactive Dashboard
Launch the dashboard to monitor surface AQI predictions and wind transport plumes:
```bash
streamlit run app.py
```
Open the printed local port URL (usually `http://localhost:8501`) in your browser to view the interactive widgets and matrices.
