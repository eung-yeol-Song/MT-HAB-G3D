"""
build_daily.py
==============
Stage 1 of the MT-HAB-G3D extension pipeline.

Converts the raw 5-minute KOEM Marine Water Quality Automatic Monitoring Network
CSV files (one per station, Korean column names) into a single tidy
*station-day* table, following the aggregation protocol described in
Section 2.2 of the manuscript:

  * daily MEAN   for  EC, water temperature, pH, DO, salinity, Chl-a, COD
  * daily MEDIAN for  turbidity  (robust to transient spikes / sensor noise)
  * a daily aggregate is produced only if >= `min_coverage` (default 0.70)
    of the 288 expected 5-min samples of that station-variable are valid
  * sentinel value -1 in the raw files is treated as MISSING
  * red-tide label  = 1 if any report within the matching radius on that day

Output
------
outputs/daily_station_day.parquet   (also .csv)
    columns: station, date, year, doy, <8 water-quality variables>,
             cov_<var> (fractional coverage), redtide_10km, redtide_30km

Usage
-----
python -m src.data.build_daily --raw_dir data/raw --out outputs/daily_station_day.parquet
"""

from __future__ import annotations

import argparse
import os
import glob
import sys
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column mapping: Korean raw header -> canonical English name used everywhere
# --------------------------------------------------------------------------- #
COLMAP = {
    "측정일시": "datetime",
    "측정소명": "station_raw",
    "전기전도도": "ec",        # electrical conductivity  [mS/cm]
    "수온": "temp",            # water temperature        [degC]
    "pH": "ph",                # pH                       [-]
    "용존산소": "do",          # dissolved oxygen         [mg/L]
    "탁도": "turbidity",       # turbidity                [NTU]
    "염분": "salinity",        # salinity                 [psu]
    "클로로필": "chl",         # chlorophyll-a            [ug/L]
    "화학적산소요구량": "cod",  # chemical oxygen demand   [mg/L]
    "10km_redtide": "redtide_10km",
    "30km_redtide": "redtide_30km",
}

# The 8 water-quality variables (graph nodes 1-8; node 9 = red tide occurrence)
WQ_VARS = ["ec", "temp", "ph", "do", "turbidity", "salinity", "chl", "cod"]

# turbidity uses the median, everything else the mean (manuscript Sec. 2.1.1)
AGG_MEDIAN = {"turbidity"}

EXPECTED_PER_DAY = 288  # 24h / 5min
SENTINEL = -1.0


def _read_station(path: str) -> pd.DataFrame:
    """Read one raw station CSV, rename columns, convert sentinels to NaN."""
    usecols = [c for c in COLMAP.keys()]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    df = df.rename(columns=COLMAP)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    for v in WQ_VARS:
        if v not in df.columns:
            df[v] = np.nan
        df[v] = pd.to_numeric(df[v], errors="coerce")
        # sentinel -1 (and any physically impossible negative) -> missing
        df.loc[df[v] <= SENTINEL, v] = np.nan

    for c in ("redtide_10km", "redtide_30km"):
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0).astype(np.int8)

    return df


def aggregate_station(df: pd.DataFrame, station: str, min_coverage: float) -> pd.DataFrame:
    """5-min -> daily aggregation with the >=70 % coverage rule."""
    df = df.copy()
    df["date"] = df["datetime"].dt.normalize()
    g = df.groupby("date", sort=True)

    out = {}
    for v in WQ_VARS:
        if v in AGG_MEDIAN:
            val = g[v].median()
        else:
            val = g[v].mean()
        cov = g[v].apply(lambda s: s.notna().sum() / EXPECTED_PER_DAY)
        # invalidate under-covered days
        val = val.where(cov >= min_coverage, np.nan)
        out[v] = val
        out[f"cov_{v}"] = cov

    out["redtide_10km"] = g["redtide_10km"].max()
    out["redtide_30km"] = g["redtide_30km"].max()

    daily = pd.DataFrame(out).reset_index()
    daily.insert(0, "station", station)
    return daily


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="directory holding the per-station CSVs")
    ap.add_argument("--out", default="outputs/daily_station_day.parquet")
    ap.add_argument("--min_coverage", type=float, default=0.70)
    ap.add_argument("--also_csv", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.csv")))
    if not files:
        sys.exit(f"no CSV found under {args.raw_dir}")

    frames = []
    for i, f in enumerate(files, 1):
        station = os.path.splitext(os.path.basename(f))[0]
        raw = _read_station(f)
        daily = aggregate_station(raw, station, args.min_coverage)
        frames.append(daily)
        print(f"[{i:2d}/{len(files)}] {station:10s} "
              f"5min={len(raw):7d} -> days={len(daily):5d} "
              f"({daily['date'].min().date()} .. {daily['date'].max().date()})  "
              f"chl_ok={daily['chl'].notna().mean():.2f}  "
              f"redtide30={daily['redtide_30km'].mean():.3f}", flush=True)
        del raw

    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["date"].dt.year
    out["doy"] = out["date"].dt.dayofyear
    out = out.sort_values(["station", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        out.to_parquet(args.out, index=False)
    except Exception:                      # pragma: no cover - pyarrow missing
        args.out = args.out.replace(".parquet", ".csv")
        out.to_csv(args.out, index=False)
    if args.also_csv:
        out.to_csv(args.out.replace(".parquet", ".csv"), index=False)

    print("\n=== summary ===")
    print(f"stations : {out['station'].nunique()}")
    print(f"rows     : {len(out)}")
    print(f"period   : {out['date'].min().date()} .. {out['date'].max().date()}")
    print("per-variable completeness:")
    print(out[WQ_VARS].notna().mean().round(3).to_string())
    print("\npositive rate by year (30 km):")
    print(out.groupby("year")["redtide_30km"].mean().round(4).to_string())
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
