"""
dataset_card.py
===============
Emits every count the revised manuscript needs to state, so that Sections 2.2,
2.3 and 4 are exactly consistent with the released code.

The numbers are recomputed from the bundle itself - not copied from a log - so
running this script is the reproducibility check for the dataset description.

Usage
-----
python -m src.dataset_card --bundle outputs/bundle_v2_paper.npz \
                           --daily outputs/daily/daily_paper.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .data.dataset import Bundle


def card(bundle_path: str, daily_path: str | None = None) -> dict:
    b = Bundle.load(bundle_path)
    cfg = b.cfg
    out: dict = {"bundle": bundle_path, "config": {
        "input_window_T": cfg.get("T"),
        "horizons_H": cfg.get("H"),
        "label_column": cfg.get("label_col"),
        "variables": list(cfg.get("variables", [])),
        "n_nodes": int(b.X.shape[2]),
        "feature_channels": int(b.X.shape[3]),
        "imputation": cfg.get("impute"),
        "max_forward_fill_days": cfg.get("max_ffill"),
        "winsor_sigma": cfg.get("winsor_sigma"),
        "correlation_threshold_tau_c": cfg.get("corr_threshold"),
        "event_radius_days": cfg.get("event_radius"),
        "train_years": cfg.get("train_years"),
        "val_years": cfg.get("val_years"),
        "test_years": cfg.get("test_years"),
    }}

    splits = {}
    for sp in ("train", "val", "test"):
        i = b.idx(sp)
        d = b.date[i]
        s = {
            "n_samples": int(len(i)),
            "n_stations": int(len(np.unique(b.station[i]))),
            "date_min": str(d.min()), "date_max": str(d.max()),
            "classification": {
                "n_labelled": int(b.m_cls[i, 0].sum()),
                "n_positive": int(b.y_cls[i, 0][b.m_cls[i, 0]].sum()),
            },
            "regression": {
                "n_chla_valid": int(b.m_reg[i, 0].sum()),
                "n_stations_with_chla": int(len(np.unique(
                    b.station[i][b.m_reg[i, 0]]))),
                "chla_mean_ugL": float(b.y_reg_raw[i, 0][b.m_reg[i, 0]].mean()),
                "chla_p90_ugL": float(np.quantile(
                    b.y_reg_raw[i, 0][b.m_reg[i, 0]], 0.9)),
                "chla_max_ugL": float(b.y_reg_raw[i, 0][b.m_reg[i, 0]].max()),
            },
        }
        ea = i[b.event_adj[i] & b.m_cls[i, 0]]
        s["event_adjacent"] = {
            "n_station_days": int(len(ea)),
            "n_positive": int(b.y_cls[ea, 0].sum()),
            "n_negative": int(len(ea) - b.y_cls[ea, 0].sum()),
            "positive_rate": (float(b.y_cls[ea, 0].mean()) if len(ea) else 0.0),
        }
        pr = s["classification"]
        s["full_year_positive_rate"] = (pr["n_positive"] / pr["n_labelled"]
                                        if pr["n_labelled"] else 0.0)
        splits[sp] = s
    out["splits"] = splits

    # per-horizon test counts (for the multi-horizon table)
    H = b.y_cls.shape[1]
    te = b.idx("test")
    out["test_by_horizon"] = {
        f"D+{h+1}": {
            "n_labelled": int(b.m_cls[te, h].sum()),
            "n_positive": int(b.y_cls[te, h][b.m_cls[te, h]].sum()),
            "n_chla_valid": int(b.m_reg[te, h].sum()),
            "event_adjacent_n": int((b.event_adj[te] & b.m_cls[te, h]).sum()),
            "event_adjacent_pos": int(
                b.y_cls[te, h][b.event_adj[te] & b.m_cls[te, h]].sum()),
        } for h in range(H)}

    # variable graph actually used
    nodes = list(cfg.get("variables", [])) + ["redtide"]
    A = np.asarray(b.A)
    edges = [(nodes[i], nodes[j]) for i in range(len(A))
             for j in range(i + 1, len(A)) if A[i, j] > 0]
    out["variable_graph"] = {
        "n_nodes": len(nodes), "node_order": nodes,
        "n_undirected_edges": len(edges),
        "density": float(A.sum() / (len(A) * (len(A) - 1))),
        "edges": [f"{u}--{v}" for u, v in edges],
    }

    if daily_path:
        d = pd.read_csv(daily_path, parse_dates=["date"])
        out["daily_table"] = {
            "path": daily_path,
            "n_station_days": int(len(d)),
            "n_stations": int(d["station"].nunique()),
            "date_min": str(d["date"].min().date()),
            "date_max": str(d["date"].max().date()),
            "redtide_positive_station_days_by_year": {
                str(k): int(v) for k, v in
                d.groupby(d["date"].dt.year)["redtide_30km"].sum().items()},
        }
    return out


def to_text(c: dict) -> str:
    s = c["splits"]
    L = []
    L.append("MANUSCRIPT NUMBERS (recomputed from the bundle)")
    L.append("=" * 62)
    cf = c["config"]
    L.append(f"Input window T = {cf['input_window_T']} days, "
             f"N = {cf['n_nodes']} nodes, C = {cf['feature_channels']} channels")
    L.append(f"Label = {cf['label_column']} (30 km matching radius); "
             f"event-adjacent radius h = {cf['event_radius_days']} days")
    L.append(f"Forward fill <= {cf['max_forward_fill_days']} days, "
             f"winsorised at +-{cf['winsor_sigma']} sd, tau_c = "
             f"{cf['correlation_threshold_tau_c']}")
    L.append("")
    L.append(f"{'split':6s} {'samples':>8s} {'stations':>9s} {'labelled':>9s} "
             f"{'pos':>5s} {'Chl-a n':>8s} {'EA n':>6s} {'EA pos':>7s}")
    for k in ("train", "val", "test"):
        v = s[k]
        L.append(f"{k:6s} {v['n_samples']:8d} {v['n_stations']:9d} "
                 f"{v['classification']['n_labelled']:9d} "
                 f"{v['classification']['n_positive']:5d} "
                 f"{v['regression']['n_chla_valid']:8d} "
                 f"{v['event_adjacent']['n_station_days']:6d} "
                 f"{v['event_adjacent']['n_positive']:7d}")
    L.append("")
    t = s["test"]
    L.append("Sentence for Section 2.3:")
    L.append(f"  \"Chl-a regression was evaluated on {t['regression']['n_chla_valid']} "
             f"station-day samples from {t['regression']['n_stations_with_chla']} "
             f"stations in 2020. The event-adjacent subset comprises "
             f"{t['event_adjacent']['n_station_days']} station-days, of which "
             f"{t['event_adjacent']['n_positive']} are positive "
             f"({100*t['event_adjacent']['positive_rate']:.1f} %), compared with a "
             f"positive rate of {100*t['full_year_positive_rate']:.2f} % on the "
             f"full-year timeline.\"")
    L.append("")
    g = c["variable_graph"]
    L.append(f"Variable graph: {g['n_nodes']} nodes, {g['n_undirected_edges']} "
             f"undirected edges, density {g['density']:.3f}")
    L.append("  " + ", ".join(g["edges"]))
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="outputs/bundle_v2_paper.npz")
    ap.add_argument("--daily", default="outputs/daily/daily_paper.csv")
    ap.add_argument("--out", default="outputs/report/dataset_card.json")
    a = ap.parse_args()
    c = card(a.bundle, a.daily)
    json.dump(c, open(a.out, "w"), indent=1)
    txt = to_text(c)
    open(a.out.replace(".json", ".txt"), "w").write(txt)
    print(txt)
