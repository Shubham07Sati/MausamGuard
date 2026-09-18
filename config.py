import os

class Config:
    # Mode selection: 'DEMO' uses synthetic physics-perturbed GFS data. 'REAL' expects actual downloaded GFS/ERA5 files.
    MODE = os.getenv("MAUSAM_MODE", "DEMO") 
    
    # Geographic Domain (India)
    LAT_MIN, LAT_MAX = 8.0, 38.0
    LON_MIN, LON_MAX = 68.0, 98.0
    GRID_RESOLUTION = 1.0 # Use 0.25 in production, 1.0 for fast MVP
    
    # Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    PLOTS_DIR = "plots"
    
    # ML Parameters
    TARGET_PERCENTILE = 0.90
    FEATURES = [
        'Lead_Time', 'Lat', 'Lon',
        'GFS_T2m', 'GFS_TP', 'GFS_Z500',
        'T2m_Anomaly', 'Z500_Anomaly', 'T2m_Temporal_Delta',
        'Z500_Grad_Lat', 'Z500_Grad_Lon', 'Z500_Grad_Mag'
    ]
