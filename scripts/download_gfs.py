"""
download_gfs.py  –  Person B / Day 1
======================================
Downloads GFS historical forecast GRIB2 files from NOAA's
public AWS S3 bucket (noaa-gfs-bdp-pds).

GFS data is organized as:
  s3://noaa-gfs-bdp-pds/gfs.<YYYYMMDD>/<HH>/atmos/gfs.t<HH>z.pgrb2.0p25.f<FFF>

Where:
  <YYYYMMDD> = Init date
  <HH>       = Init hour (00, 06, 12, 18)
  <FFF>      = Forecast hour (000, 024, 048, ... 240)

We download only the 00z initialization and forecast hours:
  f024 (Day 1) ... f240 (Day 10) at 24h steps.

Usage:
  python scripts/download_gfs.py --start 2019-06-01 --end 2019-09-30 --init-hour 00
  python scripts/download_gfs.py --years 2021 --season monsoon

Output:
  data/raw/gfs/gfs_<YYYYMMDD>_<HH>z_f<FFF>.grib2

NOTE: Each GRIB2 file is ~500MB–2GB for the full globe.
      This script applies server-side filtering via AWS S3 byte-range
      requests is NOT available for GFS GRIB2, so we download the full file
      and then subset with cfgrib/xarray in the preprocessing step.

      FALLBACK if download is slow: Use the NOAA NOMADS HTTP filter tool
      (nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl) to download
      only the variables and region you need — see download_gfs_nomads.py.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import yaml

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/gfs_download.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "india_domain.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ── AWS S3 setup (public bucket, no credentials needed) ──────────────
BUCKET = "noaa-gfs-bdp-pds"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

# ── Season helpers ────────────────────────────────────────────────────
SEASONS = {
    "monsoon":    [6, 7, 8, 9],
    "winter":     [12, 1, 2],
    "premonsoon": [3, 4, 5],
    "all":        list(range(1, 13)),
}

FORECAST_HOURS = list(range(24, 241, 24))   # [24, 48, 72, ..., 240]

# ── Build S3 key ──────────────────────────────────────────────────────
def s3_key(date: datetime, init_hour: int, fhour: int) -> str:
    yyyymmdd = date.strftime("%Y%m%d")
    hh       = f"{init_hour:02d}"
    fff      = f"{fhour:03d}"
    return f"gfs.{yyyymmdd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fff}"

# ── Download one file ─────────────────────────────────────────────────
def download_file(key: str, out_path: Path, retries: int = 3) -> bool:
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        log.info(f"  [SKIP] {out_path.name} exists ({out_path.stat().st_size/1e6:.0f} MB)")
        return True

    for attempt in range(1, retries + 1):
        try:
            log.info(f"  Downloading s3://{BUCKET}/{key}  (attempt {attempt})")
            s3.download_file(BUCKET, key, str(out_path))
            size_mb = out_path.stat().st_size / 1e6
            log.info(f"  ✓ Saved: {out_path.name}  ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            log.warning(f"  ✗ Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    log.error(f"  ✗ Failed after {retries} attempts: {key}")
    return False

# ── Date iterator ─────────────────────────────────────────────────────
def iter_dates(start: datetime, end: datetime, season_months: list[int]):
    current = start
    while current <= end:
        if current.month in season_months:
            yield current
        current += timedelta(days=1)

# ── Main ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download GFS forecasts from NOAA AWS S3.")
    p.add_argument("--start",     type=str, required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",       type=str, required=True, help="End date YYYY-MM-DD")
    p.add_argument("--season",    type=str, default="monsoon", choices=list(SEASONS.keys()))
    p.add_argument("--init-hour", type=int, default=0, choices=[0, 6, 12, 18],
                   help="GFS initialization hour (default 00z)")
    p.add_argument("--fhours",    type=int, nargs="+",
                   default=FORECAST_HOURS, help="Forecast hours to download (default 24-240)")
    p.add_argument("--dry-run",   action="store_true", help="Print S3 keys without downloading")
    return p.parse_args()

def main() -> None:
    args  = parse_args()
    cfg   = load_config()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = datetime.strptime(args.end,   "%Y-%m-%d")
    season_months = SEASONS[args.season]

    out_dir = Path(cfg["paths"]["raw_gfs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    total_dates = sum(1 for _ in iter_dates(start, end, season_months))
    total_files = total_dates * len(args.fhours)

    log.info("=" * 60)
    log.info(f"GFS Download  |  Season: {args.season}  |  Init: {args.init_hour:02d}z")
    log.info(f"Dates: {args.start} → {args.end}  |  ~{total_dates} days")
    log.info(f"Forecast hours: {args.fhours}")
    log.info(f"Estimated files: {total_files}  (~{total_files * 0.5:.0f}–{total_files * 2:.0f} GB)")
    log.info("=" * 60)

    if args.dry_run:
        log.info("[DRY RUN] Keys that would be downloaded:")
        for date in iter_dates(start, end, season_months):
            for fh in args.fhours:
                key = s3_key(date, args.init_hour, fh)
                log.info(f"  s3://{BUCKET}/{key}")
        return

    failed = []
    for date in iter_dates(start, end, season_months):
        log.info(f"\n── {date.strftime('%Y-%m-%d')} ──")
        for fh in args.fhours:
            key = s3_key(date, args.init_hour, fh)
            fname = f"gfs_{date.strftime('%Y%m%d')}_{args.init_hour:02d}z_f{fh:03d}.grib2"
            out_path = out_dir / fname
            ok = download_file(key, out_path)
            if not ok:
                failed.append(key)

    log.info(f"\n✓ Done.  Failed: {len(failed)}")
    if failed:
        fail_log = Path("logs/gfs_failed_downloads.txt")
        fail_log.write_text("\n".join(failed))
        log.info(f"  Failed keys written to {fail_log}")

if __name__ == "__main__":
    main()
