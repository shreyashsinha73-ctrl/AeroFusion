"""
INSAT-3D AOD HEATMAP GENERATOR - LOCAL (FIXED FOR 3D DATA)
For Windows, Mac, or Linux

Handles both 2D and 3D array structures
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ============================================================================
# STEP 1: DIAGNOSTIC - Examine your H5 file
# ============================================================================

def examine_h5_file(h5_file_path):
    """
    Load H5 file and show its structure
    """
    print("\n" + "=" * 80)
    print("EXAMINING YOUR H5 FILE")
    print("=" * 80)
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Show structure
            print(f"\n📁 File: {h5_file_path}")
            print(f"\n🔑 Top-level keys:")
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    print(f"   - {key}: shape={obj.shape}, dtype={obj.dtype}")
            
            # Load the actual data
            print("\n📊 Loading data...")
            
            # Load AOD
            aod = f['AOD'][:]
            lats = f['latitude'][:]
            lons = f['longitude'][:]
            
            print(f"✓ AOD shape: {aod.shape} (THIS IS KEY!)")
            print(f"✓ Latitude shape: {lats.shape}")
            print(f"✓ Longitude shape: {lons.shape}")
            
            # Statistics
            aod_float = aod.astype(float)
            
            print(f"\n📈 AOD Statistics:")
            print(f"   Min value: {float(np.nanmin(aod_float[aod_float != -999])):.0f}")
            print(f"   Max value: {float(np.nanmax(aod_float[aod_float != -999])):.0f}")
            
            # Count missing data (-999)
            missing = np.sum(aod_float == -999)
            total = aod_float.size
            coverage = (1 - missing / total) * 100
            print(f"   Coverage: {coverage:.1f}% (missing: {missing}/{total} cells)")
            
            # Test scale factors
            print(f"\n🔬 Testing scale factors:")
            aod_real = aod_float[aod_float != -999]
            
            scale_factors = [1, 1e6, 1e7, 1e8, 1e9, 1e10]
            best_scale = None
            
            for sf in scale_factors:
                scaled = aod_real / sf
                min_val = float(np.min(scaled))
                max_val = float(np.max(scaled))
                
                # AOD should be 0-5 range
                if 0 < min_val < 5 and 0 < max_val < 5:
                    print(f"   ✓ ÷ {sf:.0e}: {min_val:.4f} to {max_val:.4f} ← GOOD!")
                    if best_scale is None:
                        best_scale = sf
                else:
                    print(f"   - ÷ {sf:.0e}: {min_val:.4f} to {max_val:.4f}")
            
            print(f"\n📌 Latitude range: {float(lats.min()):.2f}°N to {float(lats.max()):.2f}°N")
            print(f"📌 Longitude range: {float(lons.min()):.2f}°E to {float(lons.max()):.2f}°E")
            
            if best_scale:
                print(f"\n✅ Recommended scale factor: {best_scale:.0e}")
            else:
                print(f"\n⚠️  No perfect scale factor found. Using 1e9 as default.")
                best_scale = 1e9
            
            return aod, lats, lons, best_scale
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# ============================================================================
# STEP 2: CREATE HEATMAP (FIXED)
# ============================================================================

def create_heatmap(h5_file_path, output_png='aod_heatmap.png', scale_factor=None):
    """
    Create professional AOD heatmap for PowerPoint
    NOW HANDLES 2D AND 3D DATA!
    """
    print("\n" + "=" * 80)
    print("GENERATING HEATMAP")
    print("=" * 80)
    
    # Examine file first
    aod, lats, lons, detected_scale = examine_h5_file(h5_file_path)
    
    if aod is None:
        return False
    
    # Use detected scale or provided scale
    if scale_factor is None:
        scale_factor = detected_scale
    
    print(f"\n🔢 Using scale factor: {scale_factor:.0e}")
    
    # ========================================================================
    # FIX: Handle different array dimensions
    # ========================================================================
    print(f"\n⚙️  Processing data...")
    print(f"   Original AOD shape: {aod.shape}")
    
    aod = aod.astype(float)
    
    # If 3D (e.g., 551 x 9 x multiple_bands), take first band
    # FIX: Handle 3D data by removing the singleton dimension
    if aod.ndim == 3:
        print(f"   ⚠️  Data is 3D! Shape: {aod.shape}")
        # This selects the first slice and removes the extra dimension
        aod = aod[0, :, :] 
        print(f"   ✓ Squeezed to 2D: {aod.shape}")
    
    # If more than 2D, flatten appropriately
    if aod.ndim > 2:
        print(f"   ⚠️  Data is {aod.ndim}D! This is unusual...")
        # Try to reshape to 2D
        try:
            aod = aod.reshape(len(lats), len(lons))
            print(f"   Reshaped to: {aod.shape}")
        except:
            print(f"   ❌ Cannot reshape data to match lat/lon dimensions")
            return False
    
    # Now process normally
    aod_scaled = aod / scale_factor
    
    # Replace -999 with NaN (missing data)
    aod_clean = np.where(aod_scaled == (-999 / scale_factor), np.nan, aod_scaled)
    
    print(f"✓ Data cleaned")
    print(f"✓ AOD shape: {aod_clean.shape}")
    print(f"✓ AOD range: {float(np.nanmin(aod_clean)):.4f} to {float(np.nanmax(aod_clean)):.4f}")
    
    # ========================================================================
    # Create mesh grid - MUST match data dimensions
    # ========================================================================
    print(f"\n🔧 Creating coordinate grids...")
    print(f"   Latitude points: {len(lats)}")
    print(f"   Longitude points: {len(lons)}")
    print(f"   AOD data shape: {aod_clean.shape}")
    
    # Ensure dimensions match
    if aod_clean.shape != (len(lats), len(lons)):
        print(f"   ⚠️  Shape mismatch! Data: {aod_clean.shape}, Expected: ({len(lats)}, {len(lons)})")
        print(f"   Attempting to fix...")
        
        # Transpose if needed
        if aod_clean.shape == (len(lons), len(lats)):
            print(f"   Transposing data...")
            aod_clean = aod_clean.T
            print(f"   Fixed shape: {aod_clean.shape}")
        else:
            print(f"   ❌ Cannot fix shape mismatch")
            return False
    
    # Create mesh grid
    try:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        print(f"✓ Grid created: {lon_grid.shape}")
    except Exception as e:
        print(f"❌ Error creating grid: {e}")
        return False
    
    # ========================================================================
    # Create figure
    # ========================================================================
    print("\n🎨 Creating visualization...")
    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
    
    try:
        # Plot contourf (filled contours)
        contour = ax.contourf(
            lon_grid, 
            lat_grid, 
            aod_clean,
            levels=20,
            cmap='RdYlGn_r',
            alpha=0.8
        )
        
        # Add contour lines
        contour_lines = ax.contour(
            lon_grid, 
            lat_grid, 
            aod_clean,
            levels=10,
            colors='black',
            alpha=0.2,
            linewidths=0.5
        )
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        print(f"✓ Contour plot created")
        
    except Exception as e:
        print(f"❌ Error creating contours: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('AOD (Aerosol Optical Depth)\nHigher = More Pollution', 
                   fontsize=11, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    ax.set_title('INSAT-3D Aerosol Optical Depth Heatmap - India', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set bounds
    ax.set_xlim(float(lons.min()), float(lons.max()))
    ax.set_ylim(float(lats.min()), float(lats.max()))
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#E8F4F8')
    
    # Add major cities
    cities = {
        'Delhi': (28.7, 77.1),
        'Mumbai': (19.1, 72.9),
        'Bangalore': (13.2, 77.6),
        'Kolkata': (22.6, 88.4),
    }
    
    for city, (lat, lon) in cities.items():
        # Only plot if in bounds
        if float(lons.min()) <= lon <= float(lons.max()) and \
           float(lats.min()) <= lat <= float(lats.max()):
            ax.plot(float(lon), float(lat), 'r*', markersize=15, 
                    markeredgecolor='black', markeredgewidth=1)
            ax.text(float(lon), float(lat)+1, city, fontsize=9, fontweight='bold', 
                    ha='center', bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='yellow', alpha=0.7))
    
    # Add legend
    legend_text = '''AOD Interpretation:
0.0-0.3 = Clear ✓
0.3-1.0 = Moderate
1.0-2.0 = Hazy
>2.0 = Very Hazy ✗'''
    
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    try:
        plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Heatmap saved: {output_png}")
        plt.close()
        
        # Show file info
        file_size = os.path.getsize(output_png) / (1024 * 1024)  # MB
        print(f"✓ File size: {file_size:.1f} MB")
        print(f"✓ Ready for PowerPoint!")
        
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          INSAT-3D AOD HEATMAP GENERATOR - LOCAL (FIXED)                      ║
║          Now handles 2D and 3D data arrays correctly!                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ======================================================================
    # EDIT THIS SECTION WITH YOUR FILE PATH
    # ======================================================================
    
    # Option 1: Set file path here
    h5_file = r"./AODSample.h5"  # YOUR PATH
    
    # Option 2: Or specify on command line
    if len(sys.argv) > 1:
        h5_file = sys.argv[1]
        print(f"Using file from command line: {h5_file}")
    
    # ======================================================================
    
    print(f"\n📂 Looking for file: {h5_file}")
    print(f"   File exists: {os.path.exists(h5_file)}")
    
    if not os.path.exists(h5_file):
        print(f"\n❌ FILE NOT FOUND!")
        sys.exit(1)
    
    # Run heatmap generation
    success = create_heatmap(h5_file, output_png='aod_heatmap_for_ppt.png')
    
    if success:
        print("\n" + "=" * 80)
        print("✅ SUCCESS!")
        print("=" * 80)
        print("\n📊 Your heatmap is ready:")
        print(f"   Location: {os.path.abspath('aod_heatmap_for_ppt.png')}")
        print(f"\n💡 Next steps:")
        print(f"   1. Open PowerPoint")
        print(f"   2. Go to Slide 8 (Expected Outcomes)")
        print(f"   3. Insert → Picture → Select 'aod_heatmap_for_ppt.png'")
        print(f"   4. Done!")
    else:
        print("\n❌ FAILED - Check error messages above")
        sys.exit(1)
