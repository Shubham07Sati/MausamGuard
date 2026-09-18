"""
download_era5.py  –  Person B / Day 1
======================================
Downloads ERA5 reanalysis data for the Indian domain using the
Copernicus Climate Data Store (CDS) API (cdsapi).

Variables downloaded:
  • 2m Temperature  (t2m)
  • Total Precipitation (tp)
  • Mean Sea Level Pressure (msl)
  • 10m U/V Wind components

Pressure-level variables (separate CDS dataset):
  • 500 hPa Geopotential Height (z500)

Usage:
  python scripts/download_era5.py --start 2019-01-01 --end 2022-12-31 --season monsoon
  python scripts/download_era5.py --start 2019-01-01 --end 2022-12-31 --season all
  python scripts/download_era5.py --years 2023 --season monsoon   # single year

Pre-requisites:
  1. pip install cdsapi
  2. Create ~/.cdsapirc with your CDS UID and API key:
       url: https://cds.climate.copernicus.eu/api/v2
       key: <UID>:<API_KEY>
     (Register free at https://cds.climate.copernicus.eu)

Output:
  data/raw/era5/era5_sfc_<YYYY>_<SEASON>.nc      ← single-level vars
  data/raw/era5/era5_pl_<YYYY>_<SEASON>.nc       ← pressure-level vars
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import cdsapi

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/era5_download.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "india_domain.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ── Season helpers ────────────────────────────────────────────────────
SEASONS = {
    "monsoon": [6, 7, 8, 9],          # JJAS
    "winter":  [12, 1, 2],            # DJF
    "all":     list(range(1, 13)),     # Full year
    "premonsoon": [3, 4, 5],          # MAM
}

def months_for_season(season: str) -> list[str]:
    months = SEASONS.get(season)
    if not months:
        raise ValueError(f"Unknown season '{season}'. Choose: {list(SEASONS.keys())}")
    return [f"{m:02d}" for m in months]

# ── Bounding box helper (CDS expects [N, W, S, E]) ────────────────────
def cds_area(cfg: dict) -> list[float]:
    d = cfg["domain"]
    return [d["lat_north"], d["lon_west"], d["lat_south"], d["lon_east"]]

# ── Surface-level download ────────────────────────────────────────────
SFC_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

def download_sfc(client: cdsapi.Client, year: int, months: list[str],
                 area: list[float], out_path: Path) -> None:
    """Download single-level ERA5 variables for a given year and months."""
    if out_path.exists():
        log.info(f"  [SKIP] {out_path.name} already exists.")
        return

    log.info(f"  Requesting SFC ERA5: year={year}, months={months}")
    request = {
        "product_type": "reanalysis",
        "variable": SFC_VARIABLES,
        "year": str(year),
        "month": months,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],   # 6-hourly
        "format": "netcdf",
        "area": area,
    }
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    log.info(f"  ✓ Saved: {out_path}")

# ── Pressure-level download ───────────────────────────────────────────
PL_VARIABLES = ["geopotential", "temperature", "u_component_of_wind",
                "v_component_of_wind", "specific_humidity"]
PRESSURE_LEVELS = ["500", "850", "200"]

def download_pl(client: cdsapi.Client, year: int, months: list[str],
                area: list[float], out_path: Path) -> None:
    """Download pressure-level ERA5 variables (Z500 etc.)."""
    if out_path.exists():
        log.info(f"  [SKIP] {out_path.name} already exists.")
        return

    log.info(f"  Requesting PL ERA5: year={year}, months={months}, levels={PRESSURE_LEVELS}")
    request = {
        "product_type": "reanalysis",
        "variable": PL_VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": str(year),
        "month": months,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "format": "netcdf",
        "area": area,
    }
    client.retrieve("reanalysis-era5-pressure-levels", request, str(out_path))
    log.info(f"  ✓ Saved: {out_path}")

# ── Main ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ERA5 reanalysis for India domain.")
    p.add_argument("--start",  type=str, help="Start date YYYY-MM-DD (overrides --years)")
    p.add_argument("--end",    type=str, help="End date YYYY-MM-DD")
    p.add_argument("--years",  type=int, nargs="+", help="Explicit year(s) e.g. 2019 2020")
    p.add_argument("--season", type=str, default="monsoon",
                   choices=list(SEASONS.keys()), help="Season to download")
    p.add_argument("--sfc-only", action="store_true", help="Only download surface variables")
    p.add_argument("--pl-only",  action="store_true", help="Only download pressure-level variables")
    return p.parse_args()

def years_from_args(args: argparse.Namespace) -> list[int]:
    if args.years:
        return args.years
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end   = datetime.strptime(args.end,   "%Y-%m-%d")
        return list(range(start.year, end.year + 1))
    raise ValueError("Provide either --start/--end or --years.")

def main() -> None:
    args   = parse_args()
    cfg    = load_config()
    area   = cds_area(cfg)
    years  = years_from_args(args)
    months = months_for_season(args.season)

    out_dir = Path(cfg["paths"]["raw_era5"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"ERA5 Download  |  Season: {args.season}  |  Years: {years}")
    log.info(f"Area (N/W/S/E): {area}  |  Months: {months}")
    log.info("=" * 60)

    client = cdsapi.Client()   # reads ~/.cdsapirc automatically

    for year in years:
        log.info(f"\n── Year {year} ──")

        if not args.pl_only:
            sfc_out = out_dir / f"era5_sfc_{year}_{args.season}.nc"
            download_sfc(client, year, months, area, sfc_out)

        if not args.sfc_only:
            pl_out = out_dir / f"era5_pl_{year}_{args.season}.nc"
            download_pl(client, year, months, area, pl_out)

    log.info("\n✓ All ERA5 downloads complete.")

if __name__ == "__main__":
    main()
