import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from config import Config

def generate_synthetic_operational_data(start_date="2023-07-01", days=30):
    """
    Generates physics-informed synthetic data.
    Instead of random uniform noise, we simulate physical model degradation:
    - Error variance increases linearly/exponentially with Lead_Time.
    - Spatial gradients of Z500 are clustered to simulate synthetic cyclones/troughs.
    """
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    print(f"[{Config.MODE} MODE] Generating physics-informed synthetic forecast data...")
    
    lats = np.arange(Config.LAT_MIN, Config.LAT_MAX + Config.GRID_RESOLUTION, Config.GRID_RESOLUTION)
    lons = np.arange(Config.LON_MIN, Config.LON_MAX + Config.GRID_RESOLUTION, Config.GRID_RESOLUTION)
    lead_times = np.arange(1, 11) # Day 1 to 10
    
    date_list = [datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=x) for x in range(days)]
    
    records = []
    
    # Pre-generate some "Storm Centers" to create realistic spatial gradients
    storm_centers = [
        {"lat": 20.0, "lon": 88.0, "date": date_list[5]}, # Bay of Bengal cyclone
        {"lat": 32.0, "lon": 76.0, "date": date_list[15]} # Western Disturbance
    ]
    
    for date in date_list:
        # Calculate distance to nearest storm center for this date
        active_storms = [s for s in storm_centers if abs((date - s["date"]).days) <= 2]
        
        for lt in lead_times:
            # Model degradation factor (error grows with lead time)
            degradation = 1.0 + (lt * 0.15) 
            
            for lat in lats:
                for lon in lons:
                    # Climatological baseline
                    base_t2m = 35.0 - (lat - 8) * 0.4  # Colder north
                    base_z500 = 5800.0 - (lat - 8) * 12
                    base_tp = np.random.exponential(2.0)
                    
                    # Storm impact (steepens gradients and increases error)
                    storm_impact = 0
                    for storm in active_storms:
                        dist = np.sqrt((lat - storm["lat"])**2 + (lon - storm["lon"])**2)
                        if dist < 5.0: # 5 degree radius
                            storm_impact += (5.0 - dist) * 2.0
                    
                    # ERA5 is Ground Truth (Base + Storm Impact)
                    era5_t2m = base_t2m - storm_impact * 0.5 
                    era5_z500 = base_z500 - storm_impact * 20.0 # Pressure drops in storms
                    era5_tp = base_tp + storm_impact * 10.0
                    
                    # GFS is Forecast (ERA5 + Error based on degradation and storm complexity)
                    # Models struggle massively around storms, especially at high lead times
                    gfs_error_scale = degradation * (1.0 + storm_impact * 0.3)
                    
                    gfs_t2m = era5_t2m + np.random.normal(0, 0.5 * gfs_error_scale)
                    gfs_z500 = era5_z500 + np.random.normal(0, 10.0 * gfs_error_scale)
                    
                    # Precipitation is notoriously hard to forecast (high variance)
                    gfs_tp = max(0, era5_tp + np.random.normal(0, 2.0 * gfs_error_scale))
                    
                    records.append({
                        "Date": date,
                        "Lead_Time": lt,
                        "Lat": lat,
                        "Lon": lon,
                        "GFS_T2m": round(gfs_t2m, 2),
                        "GFS_TP": round(gfs_tp, 2),
                        "GFS_Z500": round(gfs_z500, 2),
                        "ERA5_T2m": round(era5_t2m, 2),
                        "ERA5_TP": round(era5_tp, 2),
                        "ERA5_Z500": round(era5_z500, 2)
                    })
    
    df = pd.DataFrame(records)
    output_path = os.path.join(Config.DATA_DIR, "synthetic_merged_data.parquet")
    df.to_parquet(output_path, index=False)
    print(f"[{Config.MODE} MODE] Saved {len(df)} synthetic records to {output_path}")

if __name__ == "__main__":
    generate_synthetic_operational_data()
