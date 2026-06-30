"""
Atmospheric Transport Correction Validation Script.

This script tests the mathematical correctness, boundary condition stability,
and zero-wind exceptions of the objective 2 transport correction engine.
"""

import sys
import numpy as np
import xarray as xr
from pipeline import SatelliteFusionPipeline

def test_smearing_shift():
    """
    Test Case 1: Smearing Length Scale Shift Verification.
    Verifies that a smooth Gaussian HCHO plume shifts back by the exact mathematically
    expected distance and can be retrieved via inverse interpolation.
    """
    print("[RUNNING] Test Case 1: Smearing Length Scale shift validation...")
    pipeline = SatelliteFusionPipeline()
    
    # Place a smooth Gaussian plume centered at (18.0°N, 78.0°E)
    lat_target = 18.0
    lon_target = 78.0
    
    lat_mesh, lon_mesh = np.meshgrid(pipeline.grid_lat, pipeline.grid_lon, indexing='ij')
    # Standard deviation of 0.5 degrees ensures the peak is smooth and well-resolved
    hcho_data = 10.0 * np.exp(-((lat_mesh - lat_target)**2 / (2 * 0.5**2) + (lon_mesh - lon_target)**2 / (2 * 0.5**2)))
    
    hcho_da = xr.DataArray(
        hcho_data,
        coords=pipeline.target_coords,
        dims=['latitude', 'longitude'],
        name='hcho'
    )
    
    # Wind vector: U = 10.0 m/s (East), V = 0.0 m/s (North)
    # Lifetime: tau = 2.0 hours (7200.0 seconds)
    u_wind = 10.0
    v_wind = 0.0
    tau = 2.0
    
    # Call the displacement correction function
    corrected_da = pipeline.apply_transport_correction(
        data_da=hcho_da,
        u_wind=u_wind,
        v_wind=v_wind,
        lifetime_hours=tau,
        formula_type='physical',
        dispersion=False
    )
    
    # Mathematically expected shift in degrees longitude:
    # Ls_x = U * tau = 10.0 m/s * 7200 s = 72000.0 meters
    # lon_shift_deg = 72000.0 / (111120.0 * np.cos(np.deg2rad(lat_target)))
    # Since we shift HCHO inversely (against the wind), the plume moves West:
    lon_shift_deg = (u_wind * (tau * 3600.0)) / (111120.0 * np.cos(np.deg2rad(lat_target)))
    expected_lon_shifted = lon_target - lon_shift_deg
    
    # Interpolate corrected array at the expected shifted coordinate position
    interpolated_val = float(corrected_da.interp(latitude=lat_target, longitude=expected_lon_shifted).values)
    
    # Assert that we retrieve the peak value (10.0) within a strict tolerance (within 1%)
    assert np.abs(interpolated_val - 10.0) < 0.1, (
        f"Shift failed! Expected near 10.0 at lon {expected_lon_shifted:.4f}°, "
        f"got {interpolated_val:.4f}"
    )
    print("  - [PASS] Transport shift matches mathematical coordinate calculations.")


def test_zero_wind():
    """
    Test Case 2: Zero Wind Resilience.
    Verifies that zero wind velocities do not throw a division-by-zero exception
    and leave the grid matrix unchanged.
    """
    print("[RUNNING] Test Case 2: Zero wind velocity exception validation...")
    pipeline = SatelliteFusionPipeline()
    
    # Generate random HCHO values
    np.random.seed(42)
    hcho_data = np.random.uniform(0.0, 5.0, size=(pipeline.n_lat, pipeline.n_lon))
    hcho_da = xr.DataArray(
        hcho_data,
        coords=pipeline.target_coords,
        dims=['latitude', 'longitude'],
        name='hcho'
    )
    
    # Execute with U=0, V=0. Ensures no division by zero in degree conversion loops
    corrected_da = pipeline.apply_transport_correction(
        data_da=hcho_da,
        u_wind=0.0,
        v_wind=0.0,
        lifetime_hours=2.0,
        formula_type='physical'
    )
    
    # Check that the input and output grids are identical
    assert np.allclose(corrected_da.values, hcho_da.values, atol=1e-7, equal_nan=True), (
        "Zero wind modified the grid values!"
    )
    print("  - [PASS] Zero wind case runs without exception and preserves grid.")


def test_extreme_wind():
    """
    Test Case 3: Extreme Out-of-Bounds Wind Handling.
    Verifies that extremely high wind velocities (which shift the plume completely out
    of the Indian bounding box) do not crash the script with array boundary overflows
    and correctly return NaNs for out-of-bounds pixels.
    """
    print("[RUNNING] Test Case 3: Extreme wind boundary overflow validation...")
    pipeline = SatelliteFusionPipeline()
    
    hcho_data = np.zeros((pipeline.n_lat, pipeline.n_lon))
    lat_target = 18.0
    lon_target = 78.0
    lat_idx = np.argmin(np.abs(pipeline.grid_lat - lat_target))
    lon_idx = np.argmin(np.abs(pipeline.grid_lon - lon_target))
    hcho_data[lat_idx, lon_idx] = 10.0
    
    hcho_da = xr.DataArray(
        hcho_data,
        coords=pipeline.target_coords,
        dims=['latitude', 'longitude'],
        name='hcho'
    )
    
    # Extreme winds of 2500 m/s will shift the plume coordinates thousands of kilometers
    # out of the Indian target bounding box.
    corrected_da = pipeline.apply_transport_correction(
        data_da=hcho_da,
        u_wind=2500.0,
        v_wind=2500.0,
        lifetime_hours=5.0,
        formula_type='physical'
    )
    
    # Verify that the original peak pixel is now filled with NaN due to out-of-bounds shift
    # Use method='nearest' to handle floating-point representation in xarray select indexing
    pixel_val = corrected_da.sel(latitude=lat_target, longitude=lon_target, method='nearest').values
    assert np.isnan(pixel_val), (
        f"Expected out-of-bounds coordinate to yield NaN, but got: {pixel_val}"
    )
    print("  - [PASS] Extreme winds shift out of bounds safely, resolving to NaNs without crash.")


def main():
    print("=================================================================")
    print("   ATMOSPHERIC TRANSPORT CORRECTION: VERIFICATION ENGINE")
    print("=================================================================")
    
    tests = [test_smearing_shift, test_zero_wind, test_extreme_wind]
    successful_tests = 0
    
    for i, test in enumerate(tests, 1):
        try:
            test()
            successful_tests += 1
        except AssertionError as e:
            print(f"  - [FAIL] Test Case {i} Failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  - [CRITICAL ERROR] Test Case {i} Crashed: {e}", file=sys.stderr)
            
    print("=================================================================")
    if successful_tests == len(tests):
        print(f"Verification Passed: {successful_tests}/{len(tests)} Transport Tests Successful")
        sys.exit(0)
    else:
        print(f"Verification Failed: {successful_tests}/{len(tests)} Passed", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
