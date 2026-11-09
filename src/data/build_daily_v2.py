"""
build_daily_v2.py
=================
Stage 1 (extended).  Same coverage rules as `build_daily.py`, but

  * reads the per-station CSVs straight out of the distribution zip
    (no 1.3 GB extraction required),
  * emits **several aggregation protocols in one pass** so that the
    "daily mean vs. daily maximum" question can be answered on identical days:

        mean    arithmetic mean of the valid 5-min samples
        median  median
        max     maximum   <- the statistic actually stored in the author's
                             `gt_day` reference files (verified exactly)
        paper   manuscript Sec. 2.1.1 recipe: mean everywhere, median turbidity

  * keeps the nutrient columns (TN, TP, NH4-N, NO3-N, PO4-P, SiO2, TOC) and the
    meteorological columns, so the variable graph can be extended beyond the
    nine nodes of Table 1.

Usage
-----
python -m src.data.build_daily_v2 --zip datasets.zip --outdir outputs/daily
    -> outputs/daily/daily_mean.csv, daily_max.csv, daily_paper.csv, coverage.csv
"""

from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np
import pandas as pd

COLMAP = {
    "측정일시": "datetime",
    "전기전도도": "ec", "수온": "temp", "pH": "ph", "용존산소": "do",
    "탁도": "turbidity", "염분": "salinity", "클로로필": "chl",
    "화학적산소요구량": "cod",
    "총질소": "tn", "총인": "tp", "암모니아질소": "nh4n", "질산질소": "no3n",
    "인산인": "po4p", "규산규소": "sio2", "총유기탄소": "toc",
    "일사량": "solar", "기온": "airtemp", "풍속": "wind_speed",
    "풍향": "wind_dir", "강수량": "precip", "상대습도": "humidity",
    "10km_redtide": "redtide_10km", "30km_redtide": "redtide_30km",
}
CORE = ["ec", "temp", "ph", "do", "turbidity", "salinity", "chl", "cod"]
NUTRIENT = ["tn", "tp", "nh4n", "no3n", "po4p", "sio2", "toc"]
METEO = ["solar", "airtemp", "wind_speed", "wind_dir", "precip", "humidity"]
VARS = CORE + NUTRIENT + METEO
EXPECTED = 288


def build(zip_path: str, outdir: str, min_cov: float = 0.70):
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if n.endswith(".csv")]
    frames = {k: [] for k in ("mean", "median", "max", "paper")}
    cov_rows = []

    for i, n in enumerate(sorted(names), 1):
        station = os.path.splitext(os.path.basename(n))[0]
        df = pd.read_csv(z.open(n), usecols=lambda c: c in COLMAP, low_memory=False)
        df = df.rename(columns=COLMAP)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        for v in VARS:
            if v not in df.columns:
                df[v] = np.nan
            df[v] = pd.to_numeric(df[v], errors="coerce")
            df.loc[df[v] <= -1, v] = np.nan
        df["date"] = df["datetime"].dt.normalize()
        g = df.groupby("date", sort=True)

        cov = g[VARS].apply(lambda s: s.notna().sum() / EXPECTED)
        valid = cov >= min_cov
        aggs = {"mean": g[VARS].mean(), "median": g[VARS].median(), "max": g[VARS].max()}
        aggs["paper"] = aggs["mean"].copy()
        aggs["paper"]["turbidity"] = aggs["median"]["turbidity"]

        rt10 = g["redtide_10km"].max() if "redtide_10km" in df else None
        rt30 = g["redtide_30km"].max()
        for k, a in aggs.items():
            a = a.where(valid, np.nan).reset_index()
            a.insert(0, "station", station)
            a["redtide_30km"] = rt30.values
            if rt10 is not None:
                a["redtide_10km"] = rt10.values
            frames[k].append(a)

        c = cov.mean().to_dict()
        c["station"] = station
        cov_rows.append(c)
        print(f"[{i:2d}/{len(names)}] {station:10s} days={len(cov):5d} "
              f"core_ok={valid[CORE].mean().mean():.2f} "
              f"nutrient_ok={valid[NUTRIENT].mean().mean():.2f} "
              f"meteo_ok={valid[METEO].mean().mean():.2f}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    for k, fl in frames.items():
        out = pd.concat(fl, ignore_index=True)
        out["year"] = out["date"].dt.year
        out["doy"] = out["date"].dt.dayofyear
        out.to_csv(os.path.join(outdir, f"daily_{k}.csv"), index=False)
    cov = pd.DataFrame(cov_rows).set_index("station")
    cov.to_csv(os.path.join(outdir, "coverage.csv"))
    print("\nmean per-variable coverage over stations:")
    print(cov.mean().round(3).sort_values(ascending=False).to_string())
    print(f"\nwritten -> {outdir}/daily_[mean|median|max|paper].csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--outdir", default="outputs/daily")
    ap.add_argument("--min_cov", type=float, default=0.70)
    a = ap.parse_args()
    build(a.zip, a.outdir, a.min_cov)
