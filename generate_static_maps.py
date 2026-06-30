import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Set the path to the historical satellite dataset here.
# DuckDB will query either CSV or Parquet automatically based on the file extension.
DATA_PATH = "data/historical_satellite_data.csv"
OUTPUT_DIR = "static_maps"

# Fixed scale limits per variable for visual consistency across map comparisons
# (These defaults can be adjusted based on print_data_distribution output)
FRP_VMIN = 0.0
FRP_VMAX = 200.0

HCHO_VMIN = 0.0
HCHO_VMAX = 2.0e16


def print_data_distribution(data_path: str):
    """
    Scans the source dataset to calculate and display min, max, average, and 
    common percentiles for both FRP and HCHO to guide vmin/vmax settings.
    """
    if not os.path.exists(data_path):
        print(f"\n[INFO] Dataset not found at '{data_path}'. Skipping data distribution analysis.")
        return
        
    print(f"\n[INFO] Analyzing data distributions from '{data_path}'...")
    
    loader_func = "read_parquet" if data_path.endswith(".parquet") else "read_csv_auto"
    source = f"{loader_func}('{data_path}')"
    
    query = f"""
        SELECT 
            variable, 
            MIN(value) AS min_val, 
            MAX(value) AS max_val,
            AVG(value) AS avg_val,
            PERCENTILE_CONT(value, 0.5) AS median_val,
            PERCENTILE_CONT(value, 0.90) AS p90_val,
            PERCENTILE_CONT(value, 0.95) AS p95_val,
            PERCENTILE_CONT(value, 0.99) AS p99_val
        FROM {source}
        GROUP BY variable
    """
    try:
        df_stats = duckdb.query(query).to_df()
        print("\n=== DATASET VALUE DISTRIBUTION SUMMARY ===")
        print(df_stats.to_string(index=False))
        print("==========================================\n")
    except Exception as e:
        print(f"[WARNING] Error running distribution analysis: {e}")


def generate_static_map(year: int, season: str, variable: str, data_path: str, output_dir: str = "static_maps") -> str:
    """
    Queries/filters the dataset for the given year, season, and variable, applies 
    geographic ocean masks, plots the data with fixed ranges, and saves the output PNG.
    
    Returns:
        "SUCCESS" if file was generated successfully, or an error description string.
    """
    if not os.path.exists(data_path):
        return f"Dataset file not found at '{data_path}'"
        
    loader_func = "read_parquet" if data_path.endswith(".parquet") else "read_csv_auto"
    source = f"{loader_func}('{data_path}')"
    
    # Map season name to month filter
    # Baseline: Jan-Mar (months 1, 2, 3)
    # Burning: Oct-Nov (months 10, 11)
    if season.lower() == 'baseline':
        month_filter = "EXTRACT(month FROM CAST(date AS DATE)) IN (1, 2, 3)"
    elif season.lower() == 'burning':
        month_filter = "EXTRACT(month FROM CAST(date AS DATE)) IN (10, 11)"
    else:
        return f"Unknown season '{season}' (expected 'baseline' or 'burning')"
        
    query = f"""
        SELECT latitude, longitude, value
        FROM {source}
        WHERE EXTRACT(year FROM CAST(date AS DATE)) = {year}
          AND LOWER(variable) = '{variable.lower()}'
          AND {month_filter}
          AND latitude BETWEEN 8.0 AND 37.5
          AND longitude BETWEEN 68.0 AND 97.5
    """
    
    try:
        df = duckdb.query(query).to_df()
    except Exception as e:
        return f"DuckDB Query Error: {e}"
        
    if df.empty:
        return f"No records returned from query"
        
    # --- GEOGRAPHIC OCEAN MASK (approximate rectangular cutoffs to match app.py) ---
    arabian_sea_mask = (df['longitude'] < 72.5) & (df['latitude'] < 22.0)
    bay_of_bengal_mask = (df['longitude'] > 85.0) & (df['latitude'] < 20.0)
    df = df[~(arabian_sea_mask | bay_of_bengal_mask)].copy()
    
    if df.empty:
        return "No records remaining after applying geographic ocean mask"
        
    # Set up matplotlib plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Align styles with the dashboard's light slate theme
    fig.patch.set_facecolor('#e9ecef')
    ax.set_facecolor('#f8f9fa')
    
    # Configure variables & scales
    if variable.lower() == 'frp':
        cmap = 'YlOrRd'
        vmin, vmax = FRP_VMIN, FRP_VMAX
        title_var = "Fire Radiative Power (FRP)"
        unit_label = "FRP (MW)"
    elif variable.lower() == 'hcho':
        cmap = 'viridis'
        vmin, vmax = HCHO_VMIN, HCHO_VMAX
        title_var = "Tropospheric HCHO Plume"
        unit_label = "Column Density (molec/cm²)"
    else:
        plt.close(fig)
        return f"Unknown variable type '{variable}' (expected 'frp' or 'hcho')"
        
    # Generate scatter plot representation matching grid coordinates
    sc = ax.scatter(
        df['longitude'],
        df['latitude'],
        c=df['value'],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=12,
        alpha=0.8,
        edgecolors='none'
    )
    
    # Title & label formatting
    title_season = "Baseline Season (Jan-Mar)" if season.lower() == 'baseline' else "Biomass Burning Season (Oct-Nov)"
    ax.set_title(f"{title_var}\n{title_season} — {year}", fontsize=12, fontweight='bold', color='#212529')
    ax.set_xlabel("Longitude (°E)", color='#212529', fontsize=9)
    ax.set_ylabel("Latitude (°N)", color='#212529', fontsize=9)
    
    # Set fixed axis bounds to cover India context
    ax.set_xlim(68.0, 97.5)
    ax.set_ylim(8.0, 37.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3, color='#495057')
    ax.tick_params(colors='#212529', labelsize=8)
    
    # Add colored telemetry bar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors='#212529')
    cbar.set_label(unit_label, color='#212529', fontsize=9)
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Naming convention must match what app.py expects: {year}_{season}_{variable}.png
    output_filename = f"{year}_{season.lower()}_{variable.lower()}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return "SUCCESS"


if __name__ == "__main__":
    print("=================================================================")
    print("      HISTORICAL SATELLITE STATIC MAP GENERATION PIPELINE")
    print("=================================================================")
    
    # Step 1: Print dataset distribution first if dataset exists
    print_data_distribution(DATA_PATH)
    
    # Step 2: Loop over years, seasons, and variables
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    seasons = ['baseline', 'burning']
    variables = ['frp', 'hcho']
    
    total_expected = len(years) * len(seasons) * len(variables)
    success_count = 0
    failed_logs = []
    
    print(f"[RUNNING] Commencing map rendering loop ({total_expected} total combinations)...")
    
    for year in years:
        for season in seasons:
            for variable in variables:
                print(f" -> Rendering combination: {year} {season} {variable}...", end="", flush=True)
                result = generate_static_map(
                    year=year,
                    season=season,
                    variable=variable,
                    data_path=DATA_PATH,
                    output_dir=OUTPUT_DIR
                )
                
                if result == "SUCCESS":
                    print(" [SUCCESS]")
                    success_count += 1
                else:
                    print(f" [FAILED] ({result})")
                    failed_logs.append({
                        "key": f"{year}_{season}_{variable}",
                        "error": result
                    })
                    
    print("\n=================================================================")
    print("                       EXECUTION SUMMARY")
    print("=================================================================")
    print(f"Total combinations requested: {total_expected}")
    print(f"Successfully generated:       {success_count}")
    print(f"Failed / Missing:             {len(failed_logs)}")
    
    if failed_logs:
        print("\nFailed/Missing Combinations Details:")
        for log in failed_logs:
            print(f"  - {log['key']}: {log['error']}")
            
    print("=================================================================")
