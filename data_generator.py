import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# India Bounding Box (approximate)
LAT_MIN, LAT_MAX = 8.0, 38.0
LON_MIN, LON_MAX = 68.0, 98.0
GRID_RESOLUTION = 1.0  # 1 degree for the dummy dataset to keep it small

def generate_dummy_data(output_dir="data", start_date="2021-06-01", days=30):
    """
    Generates a dummy synthetic dataset for Person A to start building the ML pipeline.
    This simulates the merged output that Person B will eventually provide.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating dummy data for {days} days starting {start_date}...")
    
    lats = np.arange(LAT_MIN, LAT_MAX, GRID_RESOLUTION)
    lons = np.arange(LON_MIN, LON_MAX, GRID_RESOLUTION)
    lead_times = np.arange(1, 11) # Day 1 to 10
    
    date_list = [datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=x) for x in range(days)]
    
    records = []
    
    for date in date_list:
        for lt in lead_times:
            for lat in lats:
                for lon in lons:
                    # Synthetic Base Values (rough climatology approximations)
                    base_t2m = 30.0 - (lat - 8) * 0.3  # Colder as we go north
                    base_tp = np.random.exponential(5.0) # Random precipitation
                    base_z500 = 5800.0 - (lat - 8) * 10
                    
                    # Forecast vs Observation (add noise)
                    # Higher lead times should have higher errors (more variance)
                    error_scale_t2m = lt * 0.2
                    error_scale_tp = lt * 1.5
                    
                    gfs_t2m = base_t2m + np.random.normal(0, 1.0)
                    era5_t2m = gfs_t2m + np.random.normal(0, error_scale_t2m)
                    
                    gfs_tp = max(0, base_tp + np.random.normal(0, 2.0))
                    era5_tp = max(0, gfs_tp + np.random.normal(0, error_scale_tp))
                    
                    gfs_z500 = base_z500 + np.random.normal(0, 20.0)
                    era5_z500 = gfs_z500 + np.random.normal(0, lt * 5.0)
                    
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
    
    # Save to Parquet
    output_path = os.path.join(output_dir, "dummy_merged_data.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")
    print("\nSample Data:")
    print(df.head())

if __name__ == "__main__":
    generate_dummy_data()
