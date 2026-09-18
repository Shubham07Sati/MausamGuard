import pandas as pd
import numpy as np

def calculate_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the absolute error between Forecast (GFS) and Observation (ERA5).
    """
    df = df.copy()
    df['Error_T2m'] = (df['GFS_T2m'] - df['ERA5_T2m']).abs()
    df['Error_TP'] = (df['GFS_TP'] - df['ERA5_TP']).abs()
    return df

def define_bust_target(df: pd.DataFrame, variable: str = 'Error_T2m', percentile: float = 0.90) -> pd.DataFrame:
    """
    Defines a 'Bust' (1) if the error is greater than the regional (Lat/Lon) 90th percentile.
    Otherwise 'Normal' (0).
    """
    df = df.copy()
    
    # Calculate the regional threshold (90th percentile error for each Lat/Lon)
    regional_thresholds = df.groupby(['Lat', 'Lon'])[variable].transform(
        lambda x: x.quantile(percentile)
    )
    
    # Create the binary target
    target_col = f'Bust_{variable.split("_")[1]}'
    df[target_col] = (df[variable] > regional_thresholds).astype(int)
    
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts tabular features for the LightGBM model.
    """
    df = df.copy()
    
    # 1. Climatological Anomaly (rough estimate using static mean)
    df['T2m_Anomaly'] = df.groupby(['Lat', 'Lon'])['GFS_T2m'].transform(lambda x: x - x.mean())
    df['Z500_Anomaly'] = df.groupby(['Lat', 'Lon'])['GFS_Z500'].transform(lambda x: x - x.mean())
    
    # 2. Temporal Delta (Day N - Day N-1 forecast)
    df = df.sort_values(by=['Lat', 'Lon', 'Lead_Time', 'Date'])
    df['T2m_Temporal_Delta'] = df.groupby(['Lat', 'Lon', 'Lead_Time'])['GFS_T2m'].diff().fillna(0)
    
    # 3. Spatial Gradients (Difference with neighboring grid cells)
    # Sort purely by space for a given Date and Lead Time
    df = df.sort_values(by=['Date', 'Lead_Time', 'Lat', 'Lon'])
    
    # Approx Gradient North-South (Lat diff)
    df['Z500_Grad_Lat'] = df.groupby(['Date', 'Lead_Time', 'Lon'])['GFS_Z500'].diff().fillna(0)
    # Approx Gradient East-West (Lon diff)
    df['Z500_Grad_Lon'] = df.groupby(['Date', 'Lead_Time', 'Lat'])['GFS_Z500'].diff().fillna(0)
    
    # Total Gradient Magnitude
    df['Z500_Grad_Mag'] = np.sqrt(df['Z500_Grad_Lat']**2 + df['Z500_Grad_Lon']**2)
    
    return df

if __name__ == "__main__":
    # Test the pipeline on the dummy data
    print("Loading dummy data...")
    try:
        df = pd.read_parquet("data/dummy_merged_data.parquet")
        
        print("Calculating errors...")
        df = calculate_errors(df)
        
        print("Defining bust target (T2m)...")
        df = define_bust_target(df, variable='Error_T2m')
        
        print("Extracting features...")
        df = extract_features(df)
        
        print("\nFeature Engineering Complete. Sample:")
        print(df[['Date', 'Lead_Time', 'Lat', 'Lon', 'GFS_T2m', 'Error_T2m', 'Bust_T2m', 'T2m_Anomaly', 'T2m_Temporal_Delta']].head(10))
        
        # Save preprocessed data
        df.to_parquet("data/preprocessed_data.parquet", index=False)
        print("\nSaved to data/preprocessed_data.parquet")
        
    except FileNotFoundError:
        print("Dummy data not found. Run python data_generator.py first.")
