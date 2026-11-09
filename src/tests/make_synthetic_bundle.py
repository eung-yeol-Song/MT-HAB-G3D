"""
make_synthetic_bundle.py
========================
Builds a synthetic bundle with exactly the shape, dtypes, split structure and
class imbalance of the real NIFS bundle, so the whole pipeline
(train -> evaluate -> report -> figures) can be smoke-tested without the data.

    python -m src.tests.make_synthetic_bundle --out outputs/bundle_synth.npz

The numbers it produces are meaningless — it exists only to prove the code runs.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..data.dataset import Bundle, WQ_VARS, NODE_VARS, CHL_IDX, RT_IDX, build_variable_graph


def make(n_stations=9, days=2200, T=5, H=7, C=2, seed=0, pos_rate=0.014):
    rng = np.random.default_rng(seed)
    N = len(NODE_VARS)
    stations = [f"ST{i:02d}" for i in range(n_stations)]

    X, ycls, yreg, yraw, mcls, mreg, st, dates, splits, evs = [], [], [], [], [], [], [], [], [], []
    t0 = np.datetime64("2015-01-01", "D")

    for si in range(n_stations):
        # smooth autocorrelated drivers + a skewed Chl-a series with bloom peaks
        base = rng.normal(size=(days, N)).cumsum(0) / 12.0
        seas = np.sin(2 * np.pi * np.arange(days) / 365.25)[:, None]
        z = base + 0.6 * seas + rng.normal(scale=0.35, size=(days, N))
        chl_raw = np.clip(np.exp(0.55 * z[:, CHL_IDX] + 0.7) + rng.gamma(1.4, 0.8, days), 0.05, None)
        lab = (chl_raw > np.quantile(chl_raw, 1 - pos_rate * 6)).astype(np.float32)
        lab *= (rng.random(days) < 0.35)                     # thin out to a realistic rate
        z[:, RT_IDX] = lab

        mu, sd = float(chl_raw.mean()), float(chl_raw.std() or 1.0)
        chl_z = (chl_raw - mu) / sd

        for t in range(T - 1, days - H):
            d = t0 + np.timedelta64(int(t + 1), "D")
            yr = int(str(d)[:4])
            sp = "train" if yr <= 2018 else ("val" if yr == 2019 else "test")
            xc = z[t - T + 1:t + 1][:, :, None].astype(np.float32)
            if C > 1:
                xc = np.concatenate([xc, np.ones_like(xc)], axis=2)
            tgt = t + np.arange(1, H + 1)
            X.append(xc)
            ycls.append(lab[tgt].astype(np.float32))
            mcls.append(np.ones(H, bool))
            yreg.append(chl_z[tgt].astype(np.float32))
            yraw.append(chl_raw[tgt].astype(np.float32))
            mreg.append(np.ones(H, bool))
            st.append(si)
            dates.append(d)
            splits.append(sp)
            evs.append(bool(lab[max(0, t - 1):t + 3].any()))

    b = Bundle(
        X=np.stack(X), y_cls=np.stack(ycls), y_reg=np.stack(yreg),
        y_reg_raw=np.stack(yraw), m_cls=np.stack(mcls), m_reg=np.stack(mreg),
        station=np.asarray(st, np.int64), date=np.asarray(dates, "datetime64[D]"),
        split=np.asarray(splits), event_adj=np.asarray(evs), stations=stations,
        scaler={s: {"mu": {v: 0.0 for v in WQ_VARS} | {"chl": 4.0},
                    "sd": {v: 1.0 for v in WQ_VARS} | {"chl": 3.0}} for s in stations},
        cfg={"synthetic": True, "T": T, "H": H},
    )
    b.W, b.A = build_variable_graph(b, 0.30)
    return b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/bundle_synth.npz")
    ap.add_argument("--stations", type=int, default=9)
    ap.add_argument("--days", type=int, default=2200)
    a = ap.parse_args()
    b = make(n_stations=a.stations, days=a.days)
    b.save(a.out)
    print(f"X={b.X.shape}  y_cls={b.y_cls.shape}")
    for sp in ("train", "val", "test"):
        i = b.idx(sp)
        print(f"  {sp:5s} n={len(i):6d}  pos(D+1)={int(b.y_cls[i,0].sum()):4d}")
    print(f"saved -> {a.out}")
