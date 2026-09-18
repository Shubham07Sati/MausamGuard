"""
verify_downloads.py  –  Person B / Day 1
==========================================
Quick sanity check on downloaded ERA5 and GFS files.
Opens each .nc / .grib2 and prints variable names, shapes, date ranges,
and data min/max. Flags corrupt or incomplete files.

Usage:
  python scripts/verify_downloads.py --era5
  python scripts/verify_downloads.py --gfs
  python scripts/verify_downloads.py --era5 --gfs
"""

import argparse
import sys
from pathlib import Path
import yaml

# ── Logging ──────────────────────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "india_domain.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ── ERA5 verification ─────────────────────────────────────────────────
def verify_era5(cfg: dict) -> None:
    import xarray as xr
    era5_dir = Path(cfg["paths"]["raw_era5"])
    files = sorted(era5_dir.glob("*.nc"))

    if not files:
        log.warning("No ERA5 .nc files found in %s", era5_dir)
        return

    log.info(f"Found {len(files)} ERA5 file(s):")
    for f in files:
        log.info(f"\n  File: {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
        try:
            ds = xr.open_dataset(f, engine="netcdf4")
            log.info(f"    Variables  : {list(ds.data_vars)}")
            log.info(f"    Dimensions : {dict(ds.dims)}")
            if "time" in ds.coords:
                log.info(f"    Time range : {str(ds.time.values[0])[:16]} → {str(ds.time.values[-1])[:16]}")
            if "latitude" in ds.coords:
                log.info(f"    Lat range  : {float(ds.latitude.min()):.2f} → {float(ds.latitude.max()):.2f}")
                log.info(f"    Lon range  : {float(ds.longitude.min()):.2f} → {float(ds.longitude.max()):.2f}")
            # Check for NaN-only variables
            for var in ds.data_vars:
                arr = ds[var]
                if arr.isnull().all():
                    log.warning(f"    ⚠ {var}: ALL NaN!")
                else:
                    log.info(f"    {var}: min={float(arr.min()):.3f}  max={float(arr.max()):.3f}")
            ds.close()
        except Exception as e:
            log.error(f"    ✗ Failed to open: {e}")

# ── GFS verification ──────────────────────────────────────────────────
def verify_gfs(cfg: dict) -> None:
    gfs_dir = Path(cfg["paths"]["raw_gfs"])
    files = sorted(gfs_dir.glob("*.grib2"))

    if not files:
        log.warning("No GFS .grib2 files found in %s", gfs_dir)
        return

    log.info(f"Found {len(files)} GFS file(s):")
    # Try cfgrib
    try:
        import cfgrib
        import xarray as xr
    except ImportError:
        log.error("cfgrib not installed. Run: pip install cfgrib")
        return

    for f in files[:5]:   # Verify first 5 only (can take time)
        log.info(f"\n  File: {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
        try:
            datasets = cfgrib.open_datasets(str(f))
            log.info(f"    Message groups: {len(datasets)}")
            for i, ds in enumerate(datasets):
                log.info(f"    Group {i}: vars={list(ds.data_vars)}  dims={dict(ds.dims)}")
                ds.close()
        except Exception as e:
            log.error(f"    ✗ cfgrib failed: {e}")
            # Try raw size check
            size = f.stat().st_size
            if size < 10_000:
                log.error(f"    ✗ File too small ({size} bytes) — likely corrupt or 404 response")

# ── Main ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Verify downloaded ERA5 and GFS files.")
    p.add_argument("--era5", action="store_true")
    p.add_argument("--gfs",  action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = load_config()

    if not args.era5 and not args.gfs:
        log.info("No flags set. Use --era5 and/or --gfs")
        sys.exit(1)

    if args.era5:
        log.info("\n" + "=" * 50)
        log.info("ERA5 Verification")
        log.info("=" * 50)
        verify_era5(cfg)

    if args.gfs:
        log.info("\n" + "=" * 50)
        log.info("GFS Verification")
        log.info("=" * 50)
        verify_gfs(cfg)

if __name__ == "__main__":
    main()
