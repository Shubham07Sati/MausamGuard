import pandas as pd
import numpy as np
from config import Config

def calculate_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the absolute error between Forecast (GFS) and Observation (ERA5)."""
    df = df.copy()
    df['Error_T2m'] = (df['GFS_T2m'] - df['ERA5_T2m']).abs()
    df['Error_TP'] = (df['GFS_TP'] - df['ERA5_TP']).abs()
    return df

def define_bust_target(df: pd.DataFrame, variable: str = 'Error_T2m', percentile: float = 0.90) -> pd.DataFrame:
    """Defines a 'Bust' (1) if the error is greater than the regional 90th percentile."""
    df = df.copy()
    regional_thresholds = df.groupby(['Lat', 'Lon'])[variable].transform(
        lambda x: x.quantile(percentile)
    )
    target_col = f'Bust_{variable.split("_")[1]}'
    df[target_col] = (df[variable] > regional_thresholds).astype(int)
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts tabular features for the LightGBM model."""
    df = df.copy()
    
    # Handle NaNs from real GFS/ERA5 data grids (coastal boundaries, etc.)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # 1. Climatological Anomaly (rough estimate using static mean)
    df['T2m_Anomaly'] = df.groupby(['Lat', 'Lon'])['GFS_T2m'].transform(lambda x: x - x.mean())
    df['Z500_Anomaly'] = df.groupby(['Lat', 'Lon'])['GFS_Z500'].transform(lambda x: x - x.mean())
    
    # 2. Temporal Delta (Day N - Day N-1 forecast)
    df = df.sort_values(by=['Lat', 'Lon', 'Lead_Time', 'Date'])
    df['T2m_Temporal_Delta'] = df.groupby(['Lat', 'Lon', 'Lead_Time'])['GFS_T2m'].diff().fillna(0)
    
    # 3. Spatial Gradients (Difference with neighboring grid cells)
    df = df.sort_values(by=['Date', 'Lead_Time', 'Lat', 'Lon'])
    df['Z500_Grad_Lat'] = df.groupby(['Date', 'Lead_Time', 'Lon'])['GFS_Z500'].diff().fillna(0)
    df['Z500_Grad_Lon'] = df.groupby(['Date', 'Lead_Time', 'Lat'])['GFS_Z500'].diff().fillna(0)
    df['Z500_Grad_Mag'] = np.sqrt(df['Z500_Grad_Lat']**2 + df['Z500_Grad_Lon']**2)
    
    return df

if __name__ == "__main__":
    print(f"[{Config.MODE} MODE] Processing Data...")
    try:
        if Config.MODE == "DEMO":
            df = pd.read_parquet(f"{Config.DATA_DIR}/synthetic_merged_data.parquet")
        else:
            # Operational Mode expects real data joined by Person B
            df = pd.read_parquet(f"{Config.DATA_DIR}/real_merged_data.parquet")
            
        df = calculate_errors(df)
        df = define_bust_target(df, variable='Error_T2m', percentile=Config.TARGET_PERCENTILE)
        df = extract_features(df)
        
        df.to_parquet(f"{Config.DATA_DIR}/preprocessed_data.parquet", index=False)
        print(f"Saved to {Config.DATA_DIR}/preprocessed_data.parquet")
    except FileNotFoundError:
        print("Data not found. Run synthetic_gfs.py (DEMO) or Data Pipeline (REAL) first.")
