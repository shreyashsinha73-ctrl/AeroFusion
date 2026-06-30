"""
INSAT-3D AOD HEATMAP GENERATOR - MULTI-FILE COMPOSITE & MAP OUTLINES
Scans a folder for multiple H5 files, composites them to fill missing data,
and projects the heatmap onto a geographical map with India's borders.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import warnings

# Suppress warnings from calculating mean of all-NaN slices
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# Import Cartopy for mapping
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print("❌ ERROR: Cartopy library is not installed.")
    print("Please install it using: pip install cartopy (or via conda)")
    sys.exit(1)

# ============================================================================
# STEP 1: LOAD AND COMPOSITE MULTIPLE FILES
# ============================================================================

def process_h5_folder(folder_path, scale_factor=1e9):
    """
    Scans folder for .h5 files, extracts 2D AOD arrays, and creates a composite.
    """
    print("\n" + "=" * 80)
    print(f"📁 SCANNING DIRECTORY: {folder_path}")
    print("=" * 80)
    
    # Find all .h5 files
    search_pattern = os.path.join(folder_path, "*.h5")
    h5_files = glob.glob(search_pattern)
    
    if not h5_files:
        print(f"❌ No .h5 files found in {folder_path}")
        return None, None, None
        
    print(f"✅ Found {len(h5_files)} H5 files. Processing...")
    
    aod_arrays = []
    lats, lons = None, None
    
    for idx, file_path in enumerate(h5_files):
        try:
            with h5py.File(file_path, 'r') as f:
                # Grab lat/lon from the first file (assuming static grid)
                if lats is None or lons is None:
                    lats = f['latitude'][:]
                    lons = f['longitude'][:]
                
                # Extract AOD
                aod_raw = f['AOD'][:]
                aod_float = aod_raw.astype(float)
                
                # Squeeze the 3D array (1, 551, 551) down to 2D (551, 551)
                if aod_float.ndim == 3:
                    aod_float = aod_float[0, :, :]
                
                # Apply scaling and handle missing data (-999)
                aod_scaled = aod_float / scale_factor
                aod_clean = np.where(aod_scaled == (-999 / scale_factor), np.nan, aod_scaled)
                
                aod_arrays.append(aod_clean)
                print(f"  ✓ Processed [{idx+1}/{len(h5_files)}]: {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(file_path)}: {e}")
    
    if not aod_arrays:
        print("❌ Failed to extract data from any files.")
        return None, None, None
        
    # ========================================================================
    # CREATE COMPOSITE (Fill missing data)
    # ========================================================================
    print("\n🔄 Merging files to fill missing data...")
    # Stack all arrays into a 3D cube: (number_of_files, lat_points, lon_points)
    stacked_aod = np.array(aod_arrays)
    
    # Calculate the mean across the stack for each pixel, ignoring NaNs.
    # If pixel is -999 (NaN) in File A but has data in File B, it takes File B's data.
    composite_aod = np.nanmean(stacked_aod, axis=0)
    
    # Statistics
    total_pixels = composite_aod.size
    missing_pixels = np.isnan(composite_aod).sum()
    coverage = (1 - missing_pixels / total_pixels) * 100
    
    print(f"✓ Composite shape: {composite_aod.shape}")
    print(f"✓ Final Data Coverage: {coverage:.1f}%")
    print(f"✓ Min AOD: {float(np.nanmin(composite_aod)):.4f}, Max AOD: {float(np.nanmax(composite_aod)):.4f}")
    
    return composite_aod, lats, lons


# ============================================================================
# STEP 2: CREATE MAPPED HEATMAP
# ============================================================================

def create_mapped_heatmap(aod_data, lats, lons, output_png='aod_composite_map.png'):
    """
    Creates a heatmap overlayed on a geographical map of India.
    """
    print("\n🎨 Generating geographical map visualization...")
    
    # Create coordinate grid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Create figure with Cartopy map projection
    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map bounds strictly to the data's lat/lon extent
    ax.set_extent([float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())], crs=ccrs.PlateCarree())
    
    # ------------------------------------------------------------------------
    # DRAW MAP OUTLINES (INDIA BORDERS)
    # ------------------------------------------------------------------------
    print("🗺️  Drawing country borders and coastlines...")
    # Draw Coastlines
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black', zorder=3)
    # Draw Country Borders (India)
    ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black', linestyle='-', zorder=3)
    # Optional: Add faint state boundaries
    try:
        states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')
        ax.add_feature(states, edgecolor='gray', linewidth=0.5, zorder=2)
    except:
        pass # Silently skip if natural earth data download fails
    
    # Add Ocean and Land background colors
    ax.add_feature(cfeature.OCEAN, facecolor='#E8F4F8', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#F5F5F5', zorder=0)
    
    # ------------------------------------------------------------------------
    # PLOT SATELLITE DATA
    # ------------------------------------------------------------------------
    try:
        # Use pcolormesh instead of contourf to strictly follow pixel grids and better handle NaNs
        plot = ax.pcolormesh(
            lon_grid, 
            lat_grid, 
            aod_data,
            cmap='RdYlGn_r',
            alpha=0.75,
            vmin=0.0,
            vmax=2.5, # Cap visual maximum for better color scaling
            transform=ccrs.PlateCarree(),
            zorder=1
        )
        print(f"✓ Data overlay added to map")
        
    except Exception as e:
        print(f"❌ Error creating data plot: {e}")
        return False
    
    # ------------------------------------------------------------------------
    # AESTHETICS (Gridlines, Colorbar, Labels)
    # ------------------------------------------------------------------------
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 11, 'weight': 'bold'}
    gl.ylabel_style = {'size': 11, 'weight': 'bold'}
    
    # Colorbar
    cbar = plt.colorbar(plot, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label('AOD (Aerosol Optical Depth)\nHigher = More Pollution', fontsize=12, fontweight='bold')
    
    # Title
    ax.set_title('INSAT-3D Aerosol Optical Depth - Multi-File Composite', fontsize=16, fontweight='bold', pad=20)
    
    # Add major cities (Projected)
    cities = {
        'Delhi': (28.7, 77.1),
        'Mumbai': (19.1, 72.9),
        'Bangalore': (13.2, 77.6),
        'Kolkata': (22.6, 88.4),
        'Hyderabad': (17.4, 78.5)
    }
    
    for city, (lat, lon) in cities.items():
        if float(lons.min()) <= lon <= float(lons.max()) and float(lats.min()) <= lat <= float(lats.max()):
            ax.plot(lon, lat, 'k*', markersize=12, transform=ccrs.PlateCarree(), zorder=4)
            ax.text(lon, lat + 0.5, city, fontsize=9, fontweight='bold', ha='center',
                    transform=ccrs.PlateCarree(), zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add legend
    legend_text = '''AOD Interpretation:
0.0-0.3 = Clear
0.3-1.0 = Moderate
1.0-2.0 = Hazy
>2.0 = Very Hazy'''
    
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', zorder=6,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'))
    
    # Save Image
    try:
        plt.savefig(output_png, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"✓ Heatmap saved: {output_png}")
        plt.close()
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # ======================================================================
    # USER CONFIGURATION - SET FOLDER PATH HERE
    # ======================================================================
    # Provide the path to the folder containing ALL your .h5 files
    TARGET_DIRECTORY = r"./AOD_Data" 
    # ======================================================================
    
    if len(sys.argv) > 1:
        TARGET_DIRECTORY = sys.argv[1]
        
    if not os.path.exists(TARGET_DIRECTORY):
        print(f"\n❌ DIRECTORY NOT FOUND: {TARGET_DIRECTORY}")
        sys.exit(1)
        
    # 1. Process files and composite
    composite_data, lats, lons = process_h5_folder(TARGET_DIRECTORY)
    
    if composite_data is not None:
        # 2. Draw map
        success = create_mapped_heatmap(composite_data, lats, lons, output_png='aod_india_composite.png')
        
        if success:
            print("\n" + "=" * 80)
            print("✅ SUCCESS! COMPOSITE MAP GENERATED.")
            print("=" * 80)
            print(f"📊 Your presentation-ready map is saved as: {os.path.abspath('aod_india_composite.png')}")
    else:
        print("\n❌ FAILED to process the directory.")