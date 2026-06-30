"""
Production-ready Multi-Sensor Satellite Data Fusion Pipeline for Atmospheric Monitoring.

This module provides the SatelliteFusionPipeline class to ingest, clean, align,
correct, and analyze satellite-derived atmospheric columns (HCHO, NO2) and active
fire radiative power (FRP) over the Indian subcontinent.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats
import scipy.interpolate
import scipy.ndimage

class SatelliteFusionPipeline:
    """
    A class to perform multi-sensor satellite data fusion, regridding,
    meteorological transport correction, spatial correlation, and chemical attribution
    over a uniform geographic grid canvas.
    """
    def __init__(self, lat_min: float = 8.0, lat_max: float = 37.5, 
                 lon_min: float = 68.0, lon_max: float = 97.5, 
                 resolution: float = 0.1):
        """
        Initializes the geographic canvas target grid.
        
        Args:
            lat_min: Minimum latitude bound.
            lat_max: Maximum latitude bound.
            lon_min: Minimum longitude bound.
            lon_max: Maximum longitude bound.
            resolution: Grid resolution in degrees (default 0.1° ~ 11 km).
        """
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resolution = resolution
        
        # Build 1D coordinate arrays representing grid centers
        self.grid_lat = np.arange(self.lat_min, self.lat_max + self.resolution / 2.0, self.resolution)
        self.grid_lon = np.arange(self.lon_min, self.lon_max + self.resolution / 2.0, self.resolution)
        
        self.n_lat = len(self.grid_lat)
        self.n_lon = len(self.grid_lon)
        
        # Create empty template dataset for coordinate reference
        self.target_coords = {
            'latitude': self.grid_lat,
            'longitude': self.grid_lon
        }
        
        print(f"[DIAGNOSTIC] Initialized Grid Canvas over India:")
        print(f"  - Latitude range:  {self.lat_min}°N to {self.lat_max}°N")
        print(f"  - Longitude range: {self.lon_min}°E to {self.lon_max}°E")
        print(f"  - Grid Resolution: {self.resolution}° (~11 km)")
        print(f"  - Dimensions:      {self.n_lat} (Lat) x {self.n_lon} (Lon) | Total Pixels: {self.n_lat * self.n_lon}")

    def ingest_satellite_data(self, file_path: str, variable_name: str, 
                             qa_var_name: str = None, 
                             cloud_fraction_var_name: str = None, 
                             qa_threshold: float = 0.5, 
                             cloud_threshold: float = 0.4) -> xr.DataArray:
        """
        Reads raw satellite NetCDF data, applies QA/cloud filters, and regrids it.
        Supports both regular (1D coordinates) and swath (2D coordinates) NetCDFs.
        
        Args:
            file_path: Path to raw NetCDF file.
            variable_name: Name of atmospheric variable (HCHO or NO2).
            qa_var_name: Optional name of the Quality Assurance array.
            cloud_fraction_var_name: Optional name of the Cloud Fraction array.
            qa_threshold: Minimum qa_value to retain (default >= 0.5).
            cloud_threshold: Maximum cloud fraction to retain (default <= 0.4).
            
        Returns:
            Regridded xarray DataArray aligned with target canvas.
        """
        print(f"\n[DIAGNOSTIC] Ingesting '{variable_name}' from: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing satellite data file: {file_path}")
            
        try:
            ds = xr.open_dataset(file_path)
        except Exception as e:
            raise IOError(f"Failed to read NetCDF dataset from {file_path}: {e}")
            
        if variable_name not in ds:
            ds.close()
            raise KeyError(f"Variable '{variable_name}' not found in NetCDF. Available: {list(ds.keys())}")
            
        da = ds[variable_name]
        
        # 1. Identify coordinate variable names dynamically
        lat_coord = None
        lon_coord = None
        for name in ds.coords:
            if 'lat' in name.lower():
                lat_coord = name
            elif 'lon' in name.lower():
                lon_coord = name
                
        if not lat_coord or not lon_coord:
            ds.close()
            raise ValueError(f"Could not auto-detect latitude/longitude coordinates in {file_path}.")
            
        # 2. Build Quality Filter mask (qa >= 0.5 OR cloud <= 0.4)
        mask = xr.DataArray(True, coords=da.coords, dims=da.dims)
        
        qa_active = False
        cloud_active = False
        
        filter_conds = []
        if qa_var_name and qa_var_name in ds:
            qa_active = True
            filter_conds.append(ds[qa_var_name] >= qa_threshold)
            print(f"  - QA Filter configured: {qa_var_name} >= {qa_threshold}")
            
        if cloud_fraction_var_name and cloud_fraction_var_name in ds:
            cloud_active = True
            filter_conds.append(ds[cloud_fraction_var_name] <= cloud_threshold)
            print(f"  - Cloud Filter configured: {cloud_fraction_var_name} <= {cloud_threshold}")
            
        if filter_conds:
            if qa_active and cloud_active:
                mask = filter_conds[0] | filter_conds[1]
            else:
                mask = filter_conds[0]
                
            n_before = mask.size
            n_after = int(np.sum(mask.values))
            print(f"  - Filter masking: Retained {n_after} / {n_before} pixels ({n_after/n_before*100:.1f}%)")
            
        da_filtered = da.where(mask)
        
        # 3. Regrid/Interpolate onto Target Canvas
        is_2d_coords = len(ds[lat_coord].dims) > 1
        
        if is_2d_coords:
            print(f"  - 2D swath geometry detected. Performing unstructured 2D interpolation...")
            lats_flat = ds[lat_coord].values.ravel()
            lons_flat = ds[lon_coord].values.ravel()
            vals_flat = da_filtered.values.ravel()
            
            valid_idx = ~(np.isnan(lats_flat) | np.isnan(lons_flat) | np.isnan(vals_flat))
            points = np.column_stack((lons_flat[valid_idx], lats_flat[valid_idx]))
            values = vals_flat[valid_idx]
            
            if len(values) == 0:
                print("  - [WARNING] No valid satellite pixels after filtering. Returning NaN grid.")
                grid_data = np.full((self.n_lat, self.n_lon), np.nan)
            else:
                grid_lon_mesh, grid_lat_mesh = np.meshgrid(self.grid_lon, self.grid_lat)
                grid_data = scipy.interpolate.griddata(
                    points, values, (grid_lon_mesh, grid_lat_mesh),
                    method='linear', fill_value=np.nan
                )
                
            regridded_da = xr.DataArray(
                grid_data,
                coords=self.target_coords,
                dims=['latitude', 'longitude'],
                name=variable_name
            )
        else:
            print(f"  - 1D rectilinear geometry detected. Performing linear coordinate regridding...")
            da_renamed = da_filtered.rename({lat_coord: 'latitude', lon_coord: 'longitude'})
            
            regridded_da = da_renamed.interp(
                latitude=self.grid_lat,
                longitude=self.grid_lon,
                method='linear',
                kwargs={"fill_value": np.nan}
            )
            
        print(f"  - Ingestion and regridding completed. Target shape: {regridded_da.shape}")
        ds.close()
        return regridded_da

    def harmonize_fire_data(self, csv_path: str, lat_col: str = 'latitude', 
                            lon_col: str = 'longitude', frp_col: str = 'frp') -> xr.DataArray:
        """
        Ingests thermal coordinates from CSV, filters points outside boundaries,
        and bins them into target grid pixels weighted by Fire Radiative Power (FRP).
        
        Args:
            csv_path: Path to fire coordinates CSV file.
            lat_col: Latitude column name.
            lon_col: Longitude column name.
            frp_col: Fire Radiative Power (FRP) column name.
            
        Returns:
            Binned FRP xarray DataArray aligned with target canvas.
        """
        print(f"\n[DIAGNOSTIC] Harmonizing Fire data from: {os.path.basename(csv_path)}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing fire CSV file: {csv_path}")
            
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise IOError(f"Failed to read CSV from {csv_path}: {e}")
            
        required_cols = [lat_col, lon_col, frp_col]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' missing from fire CSV. Available: {list(df.columns)}")
                
        n_total = len(df)
        
        in_bounds_mask = (
            (df[lat_col] >= self.lat_min) & (df[lat_col] <= self.lat_max) &
            (df[lon_col] >= self.lon_min) & (df[lon_col] <= self.lon_max)
        )
        df_filtered = df[in_bounds_mask]
        n_filtered = len(df_filtered)
        print(f"  - Spatial filter: Retained {n_filtered} / {n_total} coordinates inside target bounding box.")
        
        # Define bin edges centered around our coordinate points
        lon_edges = np.linspace(self.lon_min - self.resolution / 2.0, self.lon_max + self.resolution / 2.0, self.n_lon + 1)
        lat_edges = np.linspace(self.lat_min - self.resolution / 2.0, self.lat_max + self.resolution / 2.0, self.n_lat + 1)
        
        # Weighted 2D spatial binning via numpy histogram
        hist, _, _ = np.histogram2d(
            df_filtered[lon_col].values,
            df_filtered[lat_col].values,
            bins=[lon_edges, lat_edges],
            weights=df_filtered[frp_col].values
        )
        
        frp_grid = hist.T
        
        frp_da = xr.DataArray(
            frp_grid,
            coords=self.target_coords,
            dims=['latitude', 'longitude'],
            name='FRP'
        )
        
        print(f"  - Binned FRP totals: {frp_grid.sum():.2f} MW across {np.sum(frp_grid > 0)} active fire grid cells.")
        return frp_da

    def cleanse_background(self, data_da: xr.DataArray, baseline: float = 4.0e15) -> xr.DataArray:
        """
        Subtracts seasonal background baseline: Net_Reactive = Observed - Baseline.
        Negative results are clipped to 0.0. NaNs are preserved.
        """
        print(f"\n[DIAGNOSTIC] Performing Seasonal Background Subtraction:")
        print(f"  - Clean-season reference baseline: {baseline:.1e} molec/cm^2")
        
        net_da = data_da - baseline
        net_da = xr.where(net_da < 0.0, 0.0, net_da)
        net_da = net_da.where(~np.isnan(data_da))
        
        valid_vals = net_da.values[~np.isnan(net_da.values)]
        if len(valid_vals) > 0:
            print(f"  - Background-subtracted statistics:")
            print(f"    * Input Mean:  {np.nanmean(data_da.values):.3e} | Net Mean:  {np.nanmean(valid_vals):.3e}")
            print(f"    * Clipped Pixels (to 0.0): {np.sum(valid_vals == 0.0)} / {len(valid_vals)} ({np.sum(valid_vals == 0.0)/len(valid_vals)*100:.1f}%)")
        else:
            print("  - [WARNING] No valid pixels to analyze after subtraction.")
            
        return net_da

    def apply_transport_correction(self, data_da: xr.DataArray, 
                                   u_wind: float, v_wind: float, 
                                   lifetime_hours: float = 2.0, 
                                   formula_type: str = 'physical',
                                   dispersion: bool = True) -> xr.DataArray:
        """
        Applies dynamic meteorological transport correction to HCHO to reverse transport smear.
        Shifts observed columns inversely back to their pyrogenic fire sources.
        """
        print(f"\n[DIAGNOSTIC] Applying Meteorological Transport Correction:")
        print(f"  - Wind components (at 850 hPa): U = {u_wind:.2f} m/s, V = {v_wind:.2f} m/s")
        print(f"  - Precursor VOC Lifetime (tau): {lifetime_hours:.2f} hours")
        print(f"  - Translation Formula Type:    '{formula_type}'")
        
        lon_mesh, lat_mesh = np.meshgrid(data_da.longitude.values, data_da.latitude.values)
        
        lat_conversion = 111120.0
        lon_conversion = 111120.0 * np.cos(np.deg2rad(lat_mesh))
        
        if formula_type == 'physical':
            tau_sec = lifetime_hours * 3600.0
            shift_x_m = u_wind * tau_sec
            shift_y_m = v_wind * tau_sec
            
            lat_shift_deg = shift_y_m / lat_conversion
            lon_shift_deg = shift_x_m / lon_conversion
        elif formula_type == 'prompt':
            shift_x_raw = u_wind / lifetime_hours
            shift_y_raw = v_wind / lifetime_hours
            
            lat_shift_deg = (shift_y_raw * 3600.0) / lat_conversion
            lon_shift_deg = (shift_x_raw * 3600.0) / lon_conversion
        else:
            raise ValueError(f"Invalid formula_type: '{formula_type}'. Choose 'physical' or 'prompt'.")
            
        mean_lat_shift = np.mean(lat_shift_deg)
        mean_lon_shift = np.mean(lon_shift_deg)
        print(f"  - Spatial displacement scale:")
        print(f"    * Lat displacement: {mean_lat_shift:.4f}° latitude ({mean_lat_shift/self.resolution:.2f} pixels)")
        print(f"    * Lon displacement: {mean_lon_shift:.4f}° longitude ({mean_lon_shift/self.resolution:.2f} pixels)")
        
        # Coordinates to query are shifted downstream (positive wind direction)
        lookup_lat = lat_mesh + lat_shift_deg
        lookup_lon = lon_mesh + lon_shift_deg
        
        lookup_rows = (lookup_lat - self.lat_min) / self.resolution
        lookup_cols = (lookup_lon - self.lon_min) / self.resolution
        
        map_coords = np.array([lookup_rows, lookup_cols])
        
        hcho_vals = data_da.values
        nan_mask = np.isnan(hcho_vals)
        clean_vals = np.where(nan_mask, 0.0, hcho_vals)
        
        shifted_vals = scipy.ndimage.map_coordinates(
            clean_vals, map_coords, order=1, mode='constant', cval=np.nan
        )
        
        valid_mask = (~nan_mask).astype(float)
        shifted_valid = scipy.ndimage.map_coordinates(
            valid_mask, map_coords, order=1, mode='constant', cval=0.0
        )
        
        shifted_vals[shifted_valid < 0.9] = np.nan
        
        # Apply physical spatial dispersion effect using a Gaussian filter
        # Sigma along X/Y scales dynamically with zonal/meridional wind speed components
        sigma_y = abs(v_wind) * 0.2
        sigma_x = abs(u_wind) * 0.2
        
        if dispersion and (sigma_x > 0.0 or sigma_y > 0.0):
            print(f"  - Applying spatial dispersion: sigma_y (lat) = {sigma_y:.3f}, sigma_x (lon) = {sigma_x:.3f}")
            # Identify NaNs before blurring to avoid NaN leakage
            nan_mask_before_blur = np.isnan(shifted_vals)
            
            # Temporarily fill NaNs with 0.0 to prevent NaN propagation during Gaussian filtering
            clean_shifted = np.where(nan_mask_before_blur, 0.0, shifted_vals)
            
            # Perform Gaussian filter blur along both grid axes
            blurred_vals = scipy.ndimage.gaussian_filter(
                clean_shifted, sigma=(sigma_y, sigma_x), mode='constant', cval=0.0
            )
            
            # Restore NaN mask to preserve QA and boundary conditions
            shifted_vals = np.where(nan_mask_before_blur, np.nan, blurred_vals)
        
        corrected_da = xr.DataArray(
            shifted_vals,
            coords=self.target_coords,
            dims=['latitude', 'longitude'],
            name=f"{data_da.name}_corrected"
        )
        
        print(f"  - Inversed wind shift applied successfully. Output canvas shape: {corrected_da.shape}")
        return corrected_da

    def compute_spatial_cross_correlation(self, data_da: xr.DataArray, 
                                           fire_da: xr.DataArray, 
                                           max_lag_pixels: int = 5) -> pd.DataFrame:
        """
        Runs a spatial cross-correlation loop (Pearson R and P-value) between
        the net enhanced columns and binned fire FRP matrix across pixel offsets.
        """
        print(f"\n[DIAGNOSTIC] Running Spatial Cross-Correlation Search (Lag range: [-{max_lag_pixels}, {max_lag_pixels}] pixels):")
        
        results = []
        data_vals = data_da.values
        fire_vals = fire_da.values
        
        for dy in range(-max_lag_pixels, max_lag_pixels + 1):
            for dx in range(-max_lag_pixels, max_lag_pixels + 1):
                shifted_data = scipy.ndimage.shift(
                    data_vals, shift=(dy, dx), order=0, mode='constant', cval=np.nan
                )
                
                valid_mask = ~(np.isnan(shifted_data) | np.isnan(fire_vals))
                valid_x = shifted_data[valid_mask]
                valid_y = fire_vals[valid_mask]
                
                n_points = len(valid_x)
                if n_points >= 3 and np.std(valid_x) > 0 and np.std(valid_y) > 0:
                    r_val, p_val = scipy.stats.pearsonr(valid_x, valid_y)
                else:
                    r_val, p_val = np.nan, np.nan
                    
                results.append({
                    'lag_y_pixels': dy,
                    'lag_x_pixels': dx,
                    'lag_latitude_deg': dy * self.resolution,
                    'lag_longitude_deg': dx * self.resolution,
                    'pearson_r': r_val,
                    'p_value': p_val,
                    'overlapping_pixels': n_points
                })
                
        df_corr = pd.DataFrame(results)
        
        valid_corr = df_corr.dropna(subset=['pearson_r'])
        if not valid_corr.empty:
            best_idx = valid_corr['pearson_r'].idxmax()
            best = valid_corr.loc[best_idx]
            print(f"  - Peak Correlation Found:")
            print(f"    * Max Pearson R:  {best['pearson_r']:.4f} (P-value: {best['p_value']:.4e})")
            print(f"    * Optimal lag:    dy = {int(best['lag_y_pixels'])} px, dx = {int(best['lag_x_pixels'])} px")
            print(f"    * Physical lag:   Lat: {best['lag_latitude_deg']:.2f}°, Lon: {best['lag_longitude_deg']:.2f}°")
            print(f"    * Overlap size:   {int(best['overlapping_pixels'])} grid cells")
        else:
            print("  - [WARNING] No overlapping grid cells could be correlated.")
            
        return df_corr

    def classify_ozone_sensitivity(self, hcho_da: xr.DataArray, no2_da: xr.DataArray) -> xr.Dataset:
        """
        Uses Formaldehyde-to-NO2 Ratio (FNR = HCHO / NO2) to classify ozone production sensitivity.
        """
        print(f"\n[DIAGNOSTIC] Running Chemical Attribution & Ozone Sensitivity Classification:")
        
        hcho_vals = hcho_da.values
        no2_vals = no2_da.values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            fnr = hcho_vals / no2_vals
            
        condlist = [
            (fnr < 1.0) & ~np.isnan(fnr),
            (fnr >= 1.0) & (fnr <= 2.0) & ~np.isnan(fnr),
            (fnr > 2.0) & ~np.isnan(fnr)
        ]
        choicelist = [0, 1, 2]
        
        classified = np.select(condlist, choicelist, default=-1)
        
        classified_float = classified.astype(float)
        classified_float[classified == -1] = np.nan
        
        fnr_da = xr.DataArray(
            fnr,
            coords=self.target_coords,
            dims=['latitude', 'longitude'],
            name='FNR'
        )
        
        sensitivity_da = xr.DataArray(
            classified_float,
            coords=self.target_coords,
            dims=['latitude', 'longitude'],
            name='Ozone_Sensitivity'
        )
        
        valid_pixels = classified_float[~np.isnan(classified_float)]
        n_valid = len(valid_pixels)
        
        if n_valid > 0:
            voc_count = np.sum(valid_pixels == 0)
            tr_count = np.sum(valid_pixels == 1)
            nox_count = np.sum(valid_pixels == 2)
            
            print(f"  - Chemical footprint statistics:")
            print(f"    * VOC-Limited (FNR < 1.0):   {voc_count} pixels ({voc_count/n_valid*100:.1f}%) [Urban footprint]")
            print(f"    * Transition (1.0 <= FNR <= 2.0): {tr_count} pixels ({tr_count/n_valid*100:.1f}%)")
            print(f"    * NOx-Limited (FNR > 2.0):   {nox_count} pixels ({nox_count/n_valid*100:.1f}%) [Pyrogenic footprint]")
        else:
            print("  - [WARNING] No valid cells to classify.")
            
        return xr.Dataset({
            'FNR': fnr_da,
            'Ozone_Sensitivity': sensitivity_da
        })

    def bundle_to_dataframe(self, hcho_da: xr.DataArray, no2_da: xr.DataArray, fire_da: xr.DataArray) -> pd.DataFrame:
        """
        Bundles aligned xarray DataArrays (FRP, HCHO, NO2, FNR) into a single Pandas DataFrame.
        """
        print("\n[DIAGNOSTIC] Bundling aligned matrices into Pandas DataFrame...")
        
        # Ensure coordinate names are standard and capitalized
        hcho = hcho_da.rename('HCHO')
        no2 = no2_da.rename('NO2')
        frp = fire_da.rename('FRP')
        
        # Calculate FNR
        with np.errstate(divide='ignore', invalid='ignore'):
            fnr_vals = hcho.values / no2.values
            
        fnr = xr.DataArray(
            fnr_vals,
            coords=hcho.coords,
            dims=hcho.dims,
            name='FNR'
        )
        
        # Merge datasets
        ds = xr.Dataset({
            'HCHO': hcho,
            'NO2': no2,
            'FRP': frp,
            'FNR': fnr
        })
        
        # Convert to DataFrame and reset index
        df = ds.to_dataframe().reset_index()
        
        # Rename coordinate columns to capitalized
        df = df.rename(columns={
            'latitude': 'Latitude',
            'longitude': 'Longitude'
        })
        print(f"  - Bundled DataFrame size: {len(df)} rows.")
        return df



def generate_synthetic_data(data_dir: str = "data"):
    """
    Generates realistic, physically consistent synthetic datasets for demonstration.
    """
    os.makedirs(data_dir, exist_ok=True)
    print(f"\n=== [DEMO SETUP] Generating Synthetic Satellite & Fire Datasets inside '{data_dir}/' ===")
    
    raw_lats = np.arange(7.95, 37.6, 0.1)
    raw_lons = np.arange(67.95, 97.6, 0.1)
    n_lat = len(raw_lats)
    n_lon = len(raw_lons)
    
    lat_mesh, lon_mesh = np.meshgrid(raw_lats, raw_lons, indexing='ij')
    
    # 1. HCHO Synthetic Dataset
    np.random.seed(100)
    hcho_bg = np.random.normal(loc=4.2e15, scale=0.08e15, size=(n_lat, n_lon))
    
    # Fire at (22.0, 80.0) wind-drifted to (22.13, 80.35)
    plume_y, plume_x = 22.13, 80.35
    hcho_plume = 1.1e16 * np.exp(-((lat_mesh - plume_y)**2 / 0.35**2 + (lon_mesh - plume_x)**2 / 0.45**2))
    
    hcho_data = hcho_bg + hcho_plume
    
    qa_vals = np.random.uniform(0.6, 1.0, size=(n_lat, n_lon))
    cloud_fracs = np.random.uniform(0.05, 0.25, size=(n_lat, n_lon))
    
    cloud_mask = lat_mesh < 12.0
    qa_vals[cloud_mask] = np.random.uniform(0.1, 0.4, size=np.sum(cloud_mask))
    cloud_fracs[cloud_mask] = np.random.uniform(0.5, 0.85, size=np.sum(cloud_mask))
    
    hcho_ds = xr.Dataset(
        data_vars={
            'hcho_column': (['latitude', 'longitude'], hcho_data),
            'qa_value': (['latitude', 'longitude'], qa_vals),
            'cloud_fraction': (['latitude', 'longitude'], cloud_fracs)
        },
        coords={
            'latitude': raw_lats,
            'longitude': raw_lons
        }
    )
    hcho_path = os.path.join(data_dir, "hcho_raw.nc")
    hcho_ds.to_netcdf(hcho_path)
    print(f"Created HCHO netCDF: {hcho_path}")
    
    # 2. NO2 Synthetic Dataset
    no2_bg = np.random.normal(loc=1.1e15, scale=0.05e15, size=(n_lat, n_lon))
    
    cities = [
        (28.6, 77.2, 8.5e15, 0.35), # New Delhi
        (19.1, 72.9, 6.5e15, 0.30), # Mumbai
        (22.6, 88.4, 5.5e15, 0.30)  # Kolkata
    ]
    city_hotspots = np.zeros_like(no2_bg)
    for c_lat, c_lon, val, scale in cities:
        city_hotspots += val * np.exp(-((lat_mesh - c_lat)**2 / scale**2 + (lon_mesh - c_lon)**2 / scale**2))
        
    fire_no2_plume = 3.0e15 * np.exp(-((lat_mesh - plume_y)**2 / 0.35**2 + (lon_mesh - plume_x)**2 / 0.45**2))
    
    no2_data = no2_bg + city_hotspots + fire_no2_plume
    
    no2_ds = xr.Dataset(
        data_vars={
            'no2_column': (['latitude', 'longitude'], no2_data),
            'qa_value': (['latitude', 'longitude'], qa_vals),
            'cloud_fraction': (['latitude', 'longitude'], cloud_fracs)
        },
        coords={
            'latitude': raw_lats,
            'longitude': raw_lons
        }
    )
    no2_path = os.path.join(data_dir, "no2_raw.nc")
    no2_ds.to_netcdf(no2_path)
    print(f"Created NO2 netCDF:  {no2_path}")
    
    # 3. Active Fire Coordinates CSV
    fire_events = []
    
    for _ in range(75):
        f_lat = np.random.normal(loc=22.0, scale=0.18)
        f_lon = np.random.normal(loc=80.0, scale=0.18)
        f_frp = np.random.exponential(scale=120.0) + 15.0
        fire_events.append({'latitude': f_lat, 'longitude': f_lon, 'frp': f_frp})
        
    out_of_bounds = [
        {'latitude': 4.5, 'longitude': 80.0, 'frp': 250.0},
        {'latitude': 41.2, 'longitude': 80.0, 'frp': 180.0},
        {'latitude': 20.0, 'longitude': 62.1, 'frp': 90.0},
        {'latitude': 20.0, 'longitude': 99.8, 'frp': 110.0}
    ]
    fire_events.extend(out_of_bounds)
    
    df_fires = pd.DataFrame(fire_events)
    fires_path = os.path.join(data_dir, "fire_events.csv")
    df_fires.to_csv(fires_path, index=False)
    print(f"Created Fires CSV:   {fires_path}")
    print("=========================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-sensor Satellite Data Fusion Pipeline for Atmospheric Monitoring over India")
    parser.add_argument('--hcho_file', type=str, help='Path to raw HCHO NetCDF file')
    parser.add_argument('--no2_file', type=str, help='Path to raw NO2 NetCDF file')
    parser.add_argument('--fire_file', type=str, help='Path to active fire coordinates CSV file')
    parser.add_argument('--u_wind', type=float, default=5.0, help='ERA5 U-wind velocity component at 850 hPa (m/s)')
    parser.add_argument('--v_wind', type=float, default=2.0, help='ERA5 V-wind velocity component at 850 hPa (m/s)')
    parser.add_argument('--voc_lifetime', type=float, default=2.0, help='Precursor VOC chemical lifetime (hours)')
    parser.add_argument('--formula', type=str, choices=['physical', 'prompt'], default='physical', 
                        help="Transport shift formula: 'physical' (Ls = wind * tau) or 'prompt' (Ls = wind / tau)")
    parser.add_argument('--output', type=str, default='output_fusion_results.nc', help='Path to save output NetCDF dataset')
    parser.add_argument('--demo', action='store_true', help='Force generation of synthetic datasets and run pipeline demonstration')
    
    args = parser.parse_args()
    
    run_demo = args.demo
    if not run_demo and (not args.hcho_file or not args.no2_file or not args.fire_file):
        print("[WARNING] Missing dataset arguments. Activating built-in pipeline Demo Mode...")
        run_demo = True
        
    if run_demo:
        demo_dir = "data"
        generate_synthetic_data(demo_dir)
        hcho_path = os.path.join(demo_dir, "hcho_raw.nc")
        no2_path = os.path.join(demo_dir, "no2_raw.nc")
        fire_path = os.path.join(demo_dir, "fire_events.csv")
    else:
        hcho_path = args.hcho_file
        no2_path = args.no2_file
        fire_path = args.fire_file

    print("=== STARTING ATMOSPHERIC DATA FUSION PIPELINE ===")
    
    try:
        pipeline = SatelliteFusionPipeline()
        
        # Objective 1: Ingestion, QA filtering, and Gridding
        hcho_regridded = pipeline.ingest_satellite_data(
            file_path=hcho_path,
            variable_name='hcho_column',
            qa_var_name='qa_value',
            cloud_fraction_var_name='cloud_fraction',
            qa_threshold=0.5,
            cloud_threshold=0.4
        )
        
        no2_regridded = pipeline.ingest_satellite_data(
            file_path=no2_path,
            variable_name='no2_column',
            qa_var_name='qa_value',
            cloud_fraction_var_name='cloud_fraction',
            qa_threshold=0.5,
            cloud_threshold=0.4
        )
        
        fire_frp_grid = pipeline.harmonize_fire_data(
            csv_path=fire_path,
            lat_col='latitude',
            lon_col='longitude',
            frp_col='frp'
        )
        
        # Objective 2: Processing, Transport Correction, and Attribution
        hcho_net = pipeline.cleanse_background(
            data_da=hcho_regridded,
            baseline=4.0e15
        )
        
        hcho_corrected = pipeline.apply_transport_correction(
            data_da=hcho_net,
            u_wind=args.u_wind,
            v_wind=args.v_wind,
            lifetime_hours=args.voc_lifetime,
            formula_type=args.formula
        )
        
        print("\n[DIAGNOSTIC] Cross-correlating Uncorrected Net HCHO with Fire FRP:")
        df_corr_uncorrected = pipeline.compute_spatial_cross_correlation(
            data_da=hcho_net,
            fire_da=fire_frp_grid,
            max_lag_pixels=5
        )
        
        print("\n[DIAGNOSTIC] Cross-correlating Wind-Corrected HCHO with Fire FRP:")
        df_corr_corrected = pipeline.compute_spatial_cross_correlation(
            data_da=hcho_corrected,
            fire_da=fire_frp_grid,
            max_lag_pixels=5
        )
        
        chemistry_ds = pipeline.classify_ozone_sensitivity(
            hcho_da=hcho_corrected,
            no2_da=no2_regridded
        )
        
        print(f"\n[DIAGNOSTIC] Saving combined datasets to: {args.output}")
        output_ds = xr.Dataset(
            data_vars={
                'Observed_HCHO': hcho_regridded,
                'Observed_NO2': no2_regridded,
                'Fire_FRP': fire_frp_grid,
                'Net_Reactive_HCHO': hcho_net,
                'Transport_Corrected_HCHO': hcho_corrected,
                'FNR_Ratio': chemistry_ds['FNR'],
                'Ozone_Sensitivity': chemistry_ds['Ozone_Sensitivity']
            }
        )
        output_ds.to_netcdf(args.output)
        print("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


AtmosphericPipeline = SatelliteFusionPipeline

if __name__ == '__main__':
    main()
