"""
build_gtday.py
==============
Stage 1b: build a station-day table from the author-supplied `gt_day` files
(datasets_24h.zip) instead of re-aggregating the 5-minute records.

Two things make this source worth a separate experiment.

1. AGGREGATION PROTOCOL.  Verified against the raw 5-min archive, every column
   of every `gt_day` file equals the daily **maximum** of that station-variable
   (mean absolute deviation = 0.000, r = 1.0000 for temp, salinity, EC, pH, DO,
   turbidity, Chl-a and COD across all five stations).  The manuscript states in
   Sec. 2.1.1 that daily *means* were used, with a *median* for turbidity.  The
   two protocols are therefore not interchangeable and the discrepancy has to be
   resolved in the text; `--agg` lets both be run through the identical model.

2. EXTENDED NODE SET.  The `gt_day` files carry nutrient and meteorological
   columns that the 9-node graph of Table 1 never uses, although the
   introduction names nutrient availability and hydrodynamic forcing as primary
   HAB drivers.  Availability at the five stations is high for nutrients
   (TN, TP ~ 0.94-0.98; NH4-N, NO3-N, PO4-P ~ 0.36-0.98) and moderate for
   meteorology (solar radiation, air temperature, wind ~ 0.38-0.76), so an
   extended-graph experiment is feasible on this subset.

Usage
-----
python -m src.data.build_gtday --raw_dir data24/gt_day --out outputs/gtday_max.csv
python -m src.data.build_daily  --raw_dir data/raw --agg max   # matched control
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

# station-name mapping between the gt_day files and the 5-min archive
STATION_MAP = {
    "Mangduk": "광양망덕", "Chonam": "광양초남", "Juklyang": "광양적량",
    "Busan": "부산수영", "Ulsan": "울산온산",
}

CORE = {                       # the 8 variables of Table 1
    "전기전도도": "ec", "수온": "temp", "pH": "ph", "용존산소": "do",
    "탁도": "turbidity", "염분": "salinity", "클로로필": "chl",
    "화학적산소요구량": "cod",
}
NUTRIENT = {                   # nutrient block (new nodes)
    "총질소": "tn", "총인": "tp", "암모니아질소": "nh4n",
    "질산질소": "no3n", "인산인": "po4p", "규산규소": "sio2",
    "총유기탄소": "toc",
}
METEO = {                      # meteorological forcing block (new nodes)
    "일사량": "solar", "기온": "airtemp", "풍속": "wind_speed",
    "풍향": "wind_dir", "강수량": "precip", "상대습도": "humidity",
}
ALL = {**CORE, **NUTRIENT, **METEO}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", default="outputs/gtday_station_day.csv")
    ap.add_argument("--min_availability", type=float, default=0.60,
                    help="drop an extended variable if fewer than this fraction of "
                         "station-days carry a value (averaged over stations)")
    a = ap.parse_args(argv)

    frames = []
    for f in sorted(glob.glob(os.path.join(a.raw_dir, "*.csv"))):
        key = os.path.splitext(os.path.basename(f))[0]
        station = STATION_MAP.get(key, key)
        d = pd.read_csv(f)
        d = d.rename(columns={"측정일시": "date", **ALL})
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        keep = ["date"] + [v for v in ALL.values() if v in d.columns] \
               + ["10km_redtide", "30km_redtide"]
        d = d[keep].replace(-1, np.nan).replace(-1.0, np.nan)
        d.insert(0, "station", station)
        d = d.rename(columns={"10km_redtide": "redtide_10km", "30km_redtide": "redtide_30km"})
        for c in ("redtide_10km", "redtide_30km"):
            d[c] = d[c].fillna(0).astype(np.int8)
        frames.append(d)
        print(f"{key:10s} -> {station:8s} rows={len(d):5d} "
              f"{d['date'].min().date()}..{d['date'].max().date()} "
              f"redtide30={int(d['redtide_30km'].sum())}")

    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["date"].dt.year
    out["doy"] = out["date"].dt.dayofyear

    av = out.groupby("station")[list(ALL.values())].apply(lambda g: g.notna().mean()).mean()
    dropped = [c for c in set(NUTRIENT.values()) | set(METEO.values())
               if c in av.index and av[c] < a.min_availability]
    print("\nmean availability per variable:")
    print(av.round(3).to_string())
    print(f"\ndropped (< {a.min_availability}): {sorted(dropped)}")
    out = out.drop(columns=dropped)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nstations={out['station'].nunique()} rows={len(out)} -> {a.out}")
    print("retained extended nodes:",
          [c for c in out.columns if c in set(NUTRIENT.values()) | set(METEO.values())])


if __name__ == "__main__":
    main()
