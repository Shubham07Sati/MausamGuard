"""
preprocess.py  –  Person B / Day 2
==================================
Preprocesses raw GFS forecasts and ERA5 ground truth into a single
flattened Parquet file for LightGBM training.

Steps:
1. Load all ERA5 NetCDF files (SFC and PL) and merge them.
2. Load all GFS GRIB2 files and concatenate them.
3. Align spatial grids (ensure Lat/Lon match exactly, handle descending/ascending).
4. Flatten the 3D data (time, lat, lon) into a 2D Tabular format.
5. Join GFS predictions with ERA5 actuals.
6. Save to data/processed/processed_training_data.parquet.

Usage:
  python scripts/preprocess.py --season monsoon
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import xarray as xr
import pandas as pd
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "india_domain.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def preprocess_era5(cfg: dict, season: str) -> xr.Dataset:
    era5_dir = Path(cfg["paths"]["raw_era5"])
    log.info("Loading ERA5 surface and pressure-level data...")
    
    # Load surface and pressure level files
    sfc_files = sorted(era5_dir.glob(f"era5_sfc_*_{season}.nc"))
    pl_files = sorted(era5_dir.glob(f"era5_pl_*_{season}.nc"))
    
    if not sfc_files or not pl_files:
        raise FileNotFoundError("Missing ERA5 files. Did you download them?")
        
    ds_sfc = xr.open_mfdataset(sfc_files, combine='by_coords')
    ds_pl = xr.open_mfdataset(pl_files, combine='by_coords')
    
    # Merge SFC and PL
    ds_era5 = xr.merge([ds_sfc, ds_pl])
    
    # Ensure latitudes are sorted consistently (ascending)
    ds_era5 = ds_era5.sortby('latitude')
    ds_era5 = ds_era5.sortby('longitude')
    
    # Rename coordinates to match GFS if needed
    ds_era5 = ds_era5.rename({'latitude': 'lat', 'longitude': 'lon'})
    return ds_era5

def preprocess_gfs(cfg: dict) -> xr.Dataset:
    gfs_dir = Path(cfg["paths"]["raw_gfs"])
    log.info("Loading GFS data...")
    
    gfs_files = sorted(gfs_dir.glob("*.grib2"))
    if not gfs_files:
        raise FileNotFoundError("Missing GFS files in data/raw/gfs/")
        
    # Using cfgrib engine. In a real scenario with many files, you might 
    # need to process in chunks to avoid memory limits.
    # Note: filter_by_keys is used to ignore problematic multidimensional variables
    datasets = []
    for f in gfs_files:
        try:
            ds = xr.open_dataset(f, engine='cfgrib', 
                                 backend_kwargs={'indexpath': ''})
            datasets.append(ds)
        except Exception as e:
            log.warning(f"Skipping {f.name} due to read error: {e}")
            
    if not datasets:
        raise RuntimeError("No valid GFS datasets loaded.")
        
    ds_gfs = xr.combine_nested(datasets, concat_dim='time')
    
    # Ensure latitudes are sorted ascending
    ds_gfs = ds_gfs.sortby('latitude')
    ds_gfs = ds_gfs.sortby('longitude')
    ds_gfs = ds_gfs.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    return ds_gfs

def create_tabular_dataset(ds_era5: xr.Dataset, ds_gfs: xr.Dataset, cfg: dict) -> pd.DataFrame:
    log.info("Flattening and aligning datasets...")
    
    # Select variables from config
    era5_vars = [v["short_name"] for v in cfg["variables"]["era5"]]
    # Ensure we only pick variables that actually exist in the dataset
    era5_vars = [v for v in era5_vars if v in ds_era5.data_vars]
    
    # Convert ERA5 to dataframe
    df_era5 = ds_era5[era5_vars].to_dataframe().reset_index()
    # Add prefix to ERA5 vars to distinguish them
    df_era5 = df_era5.rename(columns={v: f"obs_{v}" for v in era5_vars})
    
    # Similar for GFS
    # Since GFS variable names depend on level type via cfgrib, we'll just 
    # grab all data variables that are not coordinates.
    gfs_vars = list(ds_gfs.data_vars)
    df_gfs = ds_gfs[gfs_vars].to_dataframe().reset_index()
    df_gfs = df_gfs.rename(columns={v: f"fcst_{v}" for v in gfs_vars})
    
    log.info("Merging forecasts with observations...")
    # Merge on time, lat, lon
    # Note: time in GFS is initialization time + step. 
    # You'll want to join where GFS valid_time == ERA5 time.
    if 'valid_time' in df_gfs.columns:
        df_gfs = df_gfs.rename(columns={'valid_time': 'time'})
    
    # Convert time columns to same format to ensure successful merge
    df_era5['time'] = pd.to_datetime(df_era5['time'])
    df_gfs['time'] = pd.to_datetime(df_gfs['time'])
    
    df_merged = pd.merge(df_gfs, df_era5, on=['time', 'lat', 'lon'], how='inner')
    
    return df_merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=str, default="monsoon")
    args = parser.parse_args()
    
    cfg = load_config()
    
    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"processed_training_data_{args.season}.parquet"
    
    try:
        ds_era5 = preprocess_era5(cfg, args.season)
        ds_gfs = preprocess_gfs(cfg)
        
        df_final = create_tabular_dataset(ds_era5, ds_gfs, cfg)
        
        log.info(f"Saving final dataset to {out_path} ...")
        df_final.to_parquet(out_path, index=False)
        log.info(f"Done! Final dataset shape: {df_final.shape}")
        
    except Exception as e:
        log.error(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    main()
