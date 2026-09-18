"""
download_gfs_nomads.py  –  Person B / Day 1 (FALLBACK)
========================================================
FALLBACK SCRIPT: Downloads GFS data using NOAA NOMADS HTTP filter tool.
Use this if the AWS S3 approach is too slow or files are too large.

The NOMADS filter allows downloading ONLY the variables and subregion you
need, dramatically reducing file sizes (from ~500MB to ~5MB per file!).

NOMADS Filter URL format:
  https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
  ?dir=/gfs.{YYYYMMDD}/{HH}/atmos
  &file=gfs.t{HH}z.pgrb2.0p25.f{FFF}
  &var_TMP=on&var_APCP=on&var_HGT=on&var_PRMSL=on
  &lev_2_m_above_ground=on&lev_surface=on&lev_500_mb=on&lev_mean_sea_level=on
  &subregion=&leftlon=66&rightlon=98&toplat=37&bottomlat=6

NOTE: NOMADS only keeps recent data (~10 days). For historical data use
      the AWS S3 script (download_gfs.py) or GFS AWS archive.

ARCHIVE ACCESS: Historical GFS GRIB2 is on AWS:
  s3://noaa-gfs-bdp-pds/  (script above handles this)

For older data (pre-2021) use NCEI's THREDDS:
  https://www.ncei.noaa.gov/thredds/catalog/model-gfs-g4-anl-files/catalog.html
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import requests
import yaml

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/gfs_nomads_download.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "india_domain.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ── NOMADS filter base URL ────────────────────────────────────────────
NOMADS_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

SEASONS = {
    "monsoon":    [6, 7, 8, 9],
    "winter":     [12, 1, 2],
    "premonsoon": [3, 4, 5],
    "all":        list(range(1, 13)),
}

FORECAST_HOURS = list(range(24, 241, 24))

def build_url(date: datetime, init_hour: int, fhour: int, cfg: dict) -> str:
    d = cfg["domain"]
    params = {
        "dir": f"/gfs.{date.strftime('%Y%m%d')}/{init_hour:02d}/atmos",
        "file": f"gfs.t{init_hour:02d}z.pgrb2.0p25.f{fhour:03d}",
        # Variables
        "var_TMP": "on",       # Temperature
        "var_APCP": "on",      # Accumulated Precipitation
        "var_HGT": "on",       # Geopotential Height
        "var_PRMSL": "on",     # Mean Sea Level Pressure
        "var_UGRD": "on",      # U-wind
        "var_VGRD": "on",      # V-wind
        # Levels
        "lev_2_m_above_ground": "on",
        "lev_surface": "on",
        "lev_500_mb": "on",
        "lev_850_mb": "on",
        "lev_mean_sea_level": "on",
        "lev_10_m_above_ground": "on",
        # Spatial subset
        "subregion": "",
        "leftlon": str(d["lon_west"]),
        "rightlon": str(d["lon_east"]),
        "toplat": str(d["lat_north"]),
        "bottomlat": str(d["lat_south"]),
    }
    return f"{NOMADS_BASE}?{urlencode(params)}"

def download_file(url: str, out_path: Path, retries: int = 3) -> bool:
    if out_path.exists() and out_path.stat().st_size > 10_000:
        log.info(f"  [SKIP] {out_path.name}")
        return True

    for attempt in range(1, retries + 1):
        try:
            log.info(f"  GET {out_path.name}  (attempt {attempt})")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
            size_mb = out_path.stat().st_size / 1e6
            log.info(f"  ✓ {out_path.name}  ({size_mb:.2f} MB)")
            return True
        except Exception as e:
            log.warning(f"  ✗ Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)

    log.error(f"  ✗ All retries failed: {out_path.name}")
    return False

def iter_dates(start: datetime, end: datetime, season_months: list[int]):
    cur = start
    while cur <= end:
        if cur.month in season_months:
            yield cur
        cur += timedelta(days=1)

def parse_args():
    p = argparse.ArgumentParser(description="Download GFS via NOMADS HTTP filter (recent data).")
    p.add_argument("--start",     type=str, required=True)
    p.add_argument("--end",       type=str, required=True)
    p.add_argument("--season",    type=str, default="monsoon", choices=list(SEASONS.keys()))
    p.add_argument("--init-hour", type=int, default=0, choices=[0, 6, 12, 18])
    p.add_argument("--fhours",    type=int, nargs="+", default=FORECAST_HOURS)
    p.add_argument("--dry-run",   action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = load_config()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = datetime.strptime(args.end,   "%Y-%m-%d")
    season_months = SEASONS[args.season]

    out_dir = Path(cfg["paths"]["raw_gfs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"GFS NOMADS  |  Season: {args.season}  |  Init: {args.init_hour:02d}z")
    log.info("=" * 60)

    failed = []
    for date in iter_dates(start, end, season_months):
        log.info(f"\n── {date.strftime('%Y-%m-%d')} ──")
        for fh in args.fhours:
            url  = build_url(date, args.init_hour, fh, cfg)
            fname = f"gfs_{date.strftime('%Y%m%d')}_{args.init_hour:02d}z_f{fh:03d}.grib2"
            if args.dry_run:
                log.info(f"  [DRY] {url}")
                continue
            ok = download_file(url, out_dir / fname)
            if not ok:
                failed.append(url)

    if not args.dry_run:
        log.info(f"\n✓ Done.  Failed: {len(failed)}")

if __name__ == "__main__":
    main()
