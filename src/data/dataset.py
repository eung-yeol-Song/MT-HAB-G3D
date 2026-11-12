"""
dataset.py
==========
Stage 2: turn the station-day table into supervised forecasting tensors.

Implements exactly the protocol of the manuscript (Sec. 2.2, 2.3, 3.1) and adds
the machinery required by the *additional experiments*:

  * multi-horizon targets  D+1 ... D+H (direct strategy, Sec. 3.6)
  * chronological split    train 2015-2018 | val 2019 | test 2020
  * per-station standardisation with TRAIN-ONLY statistics, +-6 sigma winsorising
  * correlation variable graph estimated on TRAIN ONLY (Eq. 3-4)
  * event-adjacent evaluation subset (Eq. 1-2, radius h = 2 days)
  * optional missing-value imputation + missingness-indicator channel
    (used for the "missing-data robustness" experiment)

Tensors
-------
X       (n, T, N, C)  float32   node features, C = 1 (+1 if mask channel on)
y_cls   (n, H)        float32   red-tide occurrence at D+h
y_reg   (n, H)        float32   Chl-a at D+h, standardised
y_reg_raw (n, H)      float32   Chl-a at D+h, original ug/L
m_cls / m_reg (n, H)  bool      target-validity masks
meta    dict          station index, station name, base date, split id
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

WQ_VARS = ["ec", "temp", "ph", "do", "turbidity", "salinity", "chl", "cod"]
NODE_VARS = WQ_VARS + ["redtide"]          # N = 9 (manuscript Table 1)
CHL_IDX = NODE_VARS.index("chl")
RT_IDX = NODE_VARS.index("redtide")


# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    daily_path: str = "outputs/daily_station_day.csv"
    label_col: str = "redtide_30km"        # 30 km matching radius (manuscript)
    T: int = 5                             # input window length
    H: int = 7                             # max forecasting horizon (D+1..D+H)
    train_years: tuple = (2015, 2016, 2017, 2018)
    val_years: tuple = (2019,)
    test_years: tuple = (2020,)
    winsor_sigma: float = 6.0
    impute: str = "none"                   # {"none", "ffill"}
    max_ffill: int = 3                     # max consecutive days to carry forward
    add_mask_channel: bool = False         # extra channel flagging imputed values
    require_full_window: bool = True
    # Nodes that may be absent without discarding the sample; the corresponding
    # task simply receives mask = 0.  Chl-a is optional because the monitoring network
    # recorded NO Chl-a at all in 2015 and none at 4 stations in 2020, while those
    # station-days still carry a usable red-tide label.
    optional_nodes: tuple = ("chl",)
    corr_threshold: float = 0.30           # tau_c, tuned on validation
    knn_k: int = 0                         # >0: k-NN graph instead of the threshold
    station_knn: int = 0                   # >0: append K neighbour STATIONS as extra
                                           #     node blocks (spatial graph)
    log_target: bool = False               # model log1p(Chl-a); metrics stay in ug/L
    variables: tuple = tuple(WQ_VARS)      # node set (extendable: nutrients, meteo)
    stations: tuple = ()                   # optional station subset
    event_radius: int = 2                  # h in Eq. (1)
    seed: int = 0
    qc: str = "none"                       # {"none", "physical"}  sensor-fault screen (revision, R2-m1)
    # Extra input channels broadcast to every node (revision v13 stage-1 candidates).
    # Every model consumes them through in_channels, so the comparison stays fair.
    #   "doy"    : sin/cos of the calendar day-of-year of each window day (explicit
    #              seasonality; the day-of-year climatology is the strongest F1 row)
    #   "rthist" : red-tide history beyond the input window, known at each window
    #              day: min(days since last positive, 60)/60 and positives in the
    #              previous 30 days / 30.  Unknown-label days count as non-positive.
    extra_feats: tuple = ()

@dataclass
class Bundle:
    X: np.ndarray
    y_cls: np.ndarray
    y_reg: np.ndarray
    y_reg_raw: np.ndarray
    m_cls: np.ndarray
    m_reg: np.ndarray
    station: np.ndarray
    date: np.ndarray
    split: np.ndarray
    event_adj: np.ndarray                  # bool, member of event-adjacent subset (anchored on the D+1 target day; legacy)
    stations: list = field(default_factory=list)
    event_adj_h: np.ndarray = None         # bool (n, H): the D+h TARGET day lies within +-radius of a positive day
                                           # (revision, R2-m5: every horizon is scored on its own event window)
    scaler: dict = field(default_factory=dict)
    A: np.ndarray = None                   # binary variable adjacency (N, N)
    W: np.ndarray = None                   # |Pearson r| affinity (N, N)
    cfg: dict = field(default_factory=dict)

    def idx(self, split: str) -> np.ndarray:
        return np.where(self.split == split)[0]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X, y_cls=self.y_cls, y_reg=self.y_reg, y_reg_raw=self.y_reg_raw,
            m_cls=self.m_cls, m_reg=self.m_reg, station=self.station,
            date=self.date.astype("datetime64[D]").astype(np.int64),
            split=self.split, event_adj=self.event_adj,
            event_adj_h=(self.event_adj_h if self.event_adj_h is not None
                         else np.repeat(self.event_adj[:, None], self.y_cls.shape[1], 1)),
            stations=np.array(self.stations, dtype=object),
            A=self.A, W=self.W, cfg=json.dumps(self.cfg),
            scaler=json.dumps(self.scaler),
        )

    @staticmethod
    def load(path: str) -> "Bundle":
        z = np.load(path, allow_pickle=True)
        return Bundle(
            X=z["X"], y_cls=z["y_cls"], y_reg=z["y_reg"], y_reg_raw=z["y_reg_raw"],
            m_cls=z["m_cls"], m_reg=z["m_reg"], station=z["station"],
            date=z["date"].astype("datetime64[D]"), split=z["split"],
            event_adj=z["event_adj"], stations=list(z["stations"]),
            scaler=json.loads(str(z["scaler"])), A=z["A"], W=z["W"],
            cfg=json.loads(str(z["cfg"])),
            event_adj_h=(z["event_adj_h"] if "event_adj_h" in z.files
                         else np.repeat(z["event_adj"][:, None], z["y_cls"].shape[1], 1)),
        )


# --------------------------------------------------------------------------- #
def _split_of(year: int, cfg: DataConfig) -> str:
    if year in cfg.train_years:
        return "train"
    if year in cfg.val_years:
        return "val"
    if year in cfg.test_years:
        return "test"
    return "none"


def _event_adjacent_days(dates: np.ndarray, labels: np.ndarray, radius: int) -> np.ndarray:
    """Eq. (1)-(2): union of [t-h, t+h] around every positive day of a station."""
    flag = np.zeros(len(dates), dtype=bool)
    pos = np.where(labels == 1)[0]
    for p in pos:
        lo, hi = dates[p] - np.timedelta64(radius, "D"), dates[p] + np.timedelta64(radius, "D")
        flag |= (dates >= lo) & (dates <= hi)
    return flag


def station_neighbours(df: pd.DataFrame, stations, train_years, k: int):
    """K most related stations for each station, estimated on TRAIN YEARS ONLY.

    Why the model needs this
    ------------------------
    The variable graph connects the 9 measurements inside one station.  It is a
    weak signal: red tide's strongest variable affinity is |r| = 0.098.  The
    relationship the data actually contains is BETWEEN STATIONS - neighbouring
    sites bloom together (Nakdong-Myeongji / Nakdong-Eulsuk r = 1.00,
    Gwangyang-Mangdeok / Gwangyang-Chonam 0.91, Masan-Bongam / Masan-Samgwi 0.90)
    and a neighbour's red tide today predicts this station's red tide seven days
    out as well as the station's own history does (r = 0.48 vs 0.48), five times
    better than the whole variable graph.

    Three stations (Gwangyang-Chonam, Masan-Bongam, Masan-Yangdeok) record NO
    Chl-a at all yet do experience blooms; their neighbours, correlated at
    0.82-0.91, do have Chl-a.  Linking stations lets those sites borrow it.

    Estimated on the training years alone so no test-period co-occurrence leaks
    into the graph.
    """
    tr = df[df["year"].isin(list(train_years))]
    rt = tr.pivot_table(index="date", columns="station", values="redtide")
    ch = tr.pivot_table(index="date", columns="station", values="chl")
    nb = {}
    for s0 in stations:
        score = {}
        for s1 in stations:
            if s1 == s0:
                continue
            v = 0.0
            for piv, w in ((rt, 1.0), (ch, 0.5)):      # red tide weighted higher
                if s0 in piv and s1 in piv:
                    m = piv[s0].notna() & piv[s1].notna()
                    if m.sum() > 60 and piv[s0][m].std() > 0 and piv[s1][m].std() > 0:
                        r = float(np.corrcoef(piv[s0][m], piv[s1][m])[0, 1])
                        if np.isfinite(r):
                            v += w * abs(r)
            score[s1] = v
        nb[s0] = [s1 for s1, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]
    return nb


def block_adjacency(A_var: np.ndarray, k: int) -> np.ndarray:
    """Star-topology block adjacency for a centre station plus k neighbours.

    within every block : the variable graph
    centre <-> block j : same variable linked across stations (chl-chl, ...)
    neighbour <-> neighbour : no edge - the sample exists to forecast the centre
    """
    N = A_var.shape[0]
    M = N * (1 + k)
    A = np.zeros((M, M), np.float32)
    for b in range(1 + k):
        A[b * N:(b + 1) * N, b * N:(b + 1) * N] = A_var
    for j in range(1, 1 + k):
        for v in range(N):
            A[v, j * N + v] = 1.0
            A[j * N + v, v] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def apply_physical_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Sensor-fault screen applied to the DAILY table before standardisation (revision, R2-m1).

    The archive contains readings that are not physically plausible for coastal water
    (pH = 0, DO = 0, EC = 0 with salinity = 0, negative Chl-a). They are sensor
    outages rather than observations, and +-6 sigma winsorisation only bounds them.
    Thresholds are deliberately loose so that genuine extremes (estuarine low salinity,
    hypoxia with DO ~ 0.5 mg/L, bloom Chl-a > 100 ug/L) are kept.
    """
    df = df.copy()
    flags = {
        "ph":  (df["ph"] < 5.0) | (df["ph"] > 10.0),
        "do":  df["do"] <= 0.0,
        "ec":  df["ec"] <= 0.0,              # EC = 0 and salinity = 0 co-occur: sensor off
        "chl": df["chl"] < 0.0,
    }
    n = {}
    for v, f in flags.items():
        n[v] = int(f.sum())
        df.loc[f, v] = np.nan
        if v == "ec":
            df.loc[f, "salinity"] = np.nan
    print(f"[qc=physical] station-days set to missing: pH {n['ph']}, DO {n['do']}, "
          f"EC(+salinity) {n['ec']}, Chl-a {n['chl']}")
    return df


def build(cfg: DataConfig) -> Bundle:
    global WQ_VARS, NODE_VARS, CHL_IDX, RT_IDX
    WQ_VARS = list(cfg.variables)
    NODE_VARS = WQ_VARS + ["redtide"]
    CHL_IDX, RT_IDX = NODE_VARS.index("chl"), NODE_VARS.index("redtide")

    df = pd.read_csv(cfg.daily_path, parse_dates=["date"])
    if cfg.qc == "physical":
        df = apply_physical_qc(df)
    if cfg.stations:
        df = df[df["station"].isin(list(cfg.stations))]
    df = df[df["year"].isin(tuple(cfg.train_years) + tuple(cfg.val_years) + tuple(cfg.test_years))]
    df = df.rename(columns={cfg.label_col: "redtide"})
    stations = sorted(df["station"].unique())

    # ---------------- per-station train statistics (leak-free) -------------- #
    scaler = {}
    tr = df[df["year"].isin(cfg.train_years)]
    for s in stations:
        sub = tr[tr["station"] == s]
        mu = sub[WQ_VARS].mean()
        sd = sub[WQ_VARS].std().replace(0, np.nan)
        # stations with no training observation for a variable fall back to global
        gmu, gsd = tr[WQ_VARS].mean(), tr[WQ_VARS].std()
        mu = mu.fillna(gmu)
        sd = sd.fillna(gsd).replace(0, 1.0)
        lg = np.log1p(np.clip(sub["chl"].to_numpy(np.float64), 0, None))
        lg = lg[np.isfinite(lg)]
        scaler[s] = {"mu": mu.to_dict(), "sd": sd.to_dict(),
                     "log_mu": float(lg.mean()) if len(lg) else 0.0,
                     "log_sd": float(lg.std()) if len(lg) and lg.std() > 0 else 1.0}

    # ---------------- pass 1: one node array per station --------------------- #
    per = {}
    for s in stations:
        sub = df[df["station"] == s].sort_values("date")
        full = pd.DataFrame({"date": pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")})
        sub = full.merge(sub, on="date", how="left")

        raw = sub[WQ_VARS].to_numpy(np.float64)
        if cfg.impute == "ffill":
            filled = pd.DataFrame(raw, columns=WQ_VARS).ffill(limit=cfg.max_ffill).to_numpy()
        else:
            filled = raw
        mu = np.array([scaler[s]["mu"][v] for v in WQ_VARS])
        sd = np.array([scaler[s]["sd"][v] for v in WQ_VARS])
        z_unclipped = (filled - mu) / sd
        z = np.clip(z_unclipped, -cfg.winsor_sigma, cfg.winsor_sigma)

        lab = sub["redtide"].to_numpy(np.float64)
        lab_valid = ~np.isnan(lab)
        lab_f = np.nan_to_num(lab, nan=0.0)

        nodes = np.nan_to_num(np.concatenate([z, lab_f[:, None]], axis=1), nan=0.0)
        node_ok = np.concatenate([~np.isnan(z), lab_valid[:, None]], axis=1)

        chl_raw = filled[:, WQ_VARS.index("chl")]
        if cfg.log_target:
            # Chl-a is strongly right-skewed (train mean 6.9 ug/L, blooms > 25).
            # Fitting log1p and inverting for the metrics keeps the loss from
            # being dominated by a handful of peaks.  y_reg_raw stays in ug/L so
            # every reported number is still on the physical scale.
            lg = np.log1p(np.clip(chl_raw, 0, None))
            lmu, lsd = scaler[s]["log_mu"], scaler[s]["log_sd"]
            chl_z = (lg - lmu) / lsd
        else:
            chl_z = z_unclipped[:, WQ_VARS.index("chl")]

        # ---- extra channels (revision v13): computed per calendar day, causal ----
        extra = []
        if "doy" in cfg.extra_feats:
            doy = sub["date"].dt.dayofyear.to_numpy(np.float64)
            ang = 2.0 * np.pi * (doy - 1.0) / 365.25
            extra += [np.sin(ang), np.cos(ang)]
        if "rthist" in cfg.extra_feats:
            posday = (lab_valid & (lab_f > 0.5)).astype(np.int64)
            since = np.full(len(posday), 60.0)
            last = None
            for i in range(len(posday)):
                if posday[i]:
                    last = i          # day i itself counts: "since" is 0 on a positive day
                if last is not None:
                    since[i] = min(float(i - last), 60.0)
            csum = np.concatenate([[0], np.cumsum(posday)])
            n30 = np.array([csum[i + 1] - csum[max(0, i + 1 - 30)] for i in range(len(posday))],
                           np.float64)
            extra += [since / 60.0, n30 / 30.0]
        extra = np.stack(extra, axis=1).astype(np.float32) if extra else np.zeros((len(sub), 0), np.float32)

        per[s] = dict(
            dates=sub["date"].to_numpy("datetime64[D]"),
            years=sub["date"].dt.year.to_numpy(),
            nodes=nodes, ok=node_ok, chl_raw=chl_raw, chl_z=chl_z,
            lab_f=lab_f, lab_valid=lab_valid, extra=extra,
            ev=_event_adjacent_days(sub["date"].to_numpy("datetime64[D]"),
                                    np.where(lab_valid, lab_f, 0), cfg.event_radius),
            pos={d: i for i, d in enumerate(sub["date"].to_numpy("datetime64[D]"))})

    # ---------------- station graph (train-only) ----------------------------- #
    K = int(cfg.station_knn or 0)
    nbrs = station_neighbours(df, stations, cfg.train_years, K) if K > 0 else {s: [] for s in stations}
    if K > 0:
        print(f"station graph: each station linked to its {K} closest neighbours "
              f"(estimated on {list(cfg.train_years)})")
        for s in stations[:5]:
            print(f"   {s} -> {', '.join(nbrs[s])}")

    Xs, ycs, yrs, yrraw, mcs, mrs, st_ix, dts, sps, evs = [], [], [], [], [], [], [], [], [], []
    evs_h = []                                            # per-horizon event membership (revision)
    NV = len(NODE_VARS)

    # ---------------- pass 2: windows, with neighbour blocks ----------------- #
    for si, s in enumerate(stations):
        P = per[s]
        D = len(P["dates"])
        T, H = cfg.T, cfg.H
        required_nodes = np.array([n not in cfg.optional_nodes for n in NODE_VARS])
        for t in range(T - 1, D - H):
            win = slice(t - T + 1, t + 1)
            ok = P["ok"][win]
            if cfg.require_full_window and not ok[:, required_nodes].all():
                continue
            sp = _split_of(P["years"][t + 1], cfg)
            if sp == "none":
                continue

            blocks_v = [P["nodes"][win]]                      # (T, NV)
            blocks_m = [ok.astype(np.float32)]
            wdates = P["dates"][win]
            for nb in nbrs[s]:
                Q = per[nb]
                bv = np.zeros((T, NV), np.float32)
                bm = np.zeros((T, NV), np.float32)
                for ti, dd in enumerate(wdates):
                    j = Q["pos"].get(dd)
                    if j is not None:
                        bv[ti] = Q["nodes"][j]
                        bm[ti] = Q["ok"][j]
                blocks_v.append(bv)
                blocks_m.append(bm)

            xv = np.concatenate(blocks_v, axis=1)             # (T, NV*(1+K))
            xc = xv[:, :, None].astype(np.float32)
            if cfg.add_mask_channel:
                xm = np.concatenate(blocks_m, axis=1)[:, :, None].astype(np.float32)
                xc = np.concatenate([xc, xm], axis=2)
            if P["extra"].shape[1]:
                # (T, F) -> (T, N_total, F): the same calendar/history value on every node
                xe = np.repeat(P["extra"][win][:, None, :], xv.shape[1], axis=1)
                xc = np.concatenate([xc, xe], axis=2)

            hz = np.arange(1, H + 1)
            tgt = t + hz
            yc = P["lab_f"][tgt]
            mc = P["lab_valid"][tgt]
            yr = P["chl_z"][tgt]
            yraw = P["chl_raw"][tgt]
            mr = (~np.isnan(P["chl_raw"][tgt])) & bool(ok[:, CHL_IDX].all())
            if not (mc[0] or mr[0]):
                continue

            Xs.append(xc)
            ycs.append(yc.astype(np.float32))
            mcs.append(mc)
            yrs.append(np.nan_to_num(yr, nan=0.0).astype(np.float32))
            yrraw.append(np.nan_to_num(yraw, nan=0.0).astype(np.float32))
            mrs.append(mr)
            st_ix.append(si)
            dts.append(P["dates"][t + 1])
            sps.append(sp)
            evs.append(bool(P["ev"][t + 1]))
            evs_h.append([bool(P["ev"][t + h]) for h in hz])    # t + H < D by the loop bound

    b = Bundle(
        X=np.stack(Xs), y_cls=np.stack(ycs), y_reg=np.stack(yrs),
        y_reg_raw=np.stack(yrraw), m_cls=np.stack(mcs), m_reg=np.stack(mrs),
        station=np.asarray(st_ix, np.int64), date=np.asarray(dts, "datetime64[D]"),
        split=np.asarray(sps), event_adj=np.asarray(evs), stations=stations,
        scaler=scaler, cfg=asdict(cfg), event_adj_h=np.asarray(evs_h, dtype=bool),
    )
    NV = len(NODE_VARS)
    Wv, Av = build_variable_graph(b, cfg.corr_threshold, getattr(cfg, "knn_k", 0), NV)
    if K > 0:
        b.A = block_adjacency(Av, K)
        b.W = np.pad(Wv, ((0, NV * K), (0, NV * K)))
    else:
        b.W, b.A = Wv, Av
    b.cfg["station_knn"] = K
    b.cfg["log_target"] = bool(cfg.log_target)
    b.cfg["node_vars"] = NODE_VARS
    return b


# --------------------------------------------------------------------------- #
def knn_graph(W: np.ndarray, k: int) -> np.ndarray:
    """Connect every node to its k strongest correlates, then symmetrise.

    Why this exists
    ---------------
    The thresholded construction (|r| >= tau) leaves a node isolated whenever a
    variable happens to correlate weakly with everything else.  On this dataset,
    at the configured tau = 0.30, that hits BOTH forecast targets: Chl-a's
    strongest affinity is 0.288 (to DO) and red-tide's is 0.098 (to water
    temperature), so both fall under the cut and end up with degree 0.  The
    graph convolution then has no edge along which to move information into or
    out of the two variables the model is asked to predict, which makes the
    graph pathway inert and reduces MT-HAB-G3D to a plain sequence encoder.

    k-NN guarantees a minimum degree of k for every node.  It is the standard
    construction in the graph-forecasting literature (MTGNN and relatives).
    """
    N = W.shape[0]
    A = np.zeros_like(W)
    for i in range(N):
        order = np.argsort(-W[i])
        c = 0
        for j in order:
            if j != i and W[i, j] > 0:
                A[i, j] = 1.0
                c += 1
            if c >= k:
                break
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float32)


def build_variable_graph(b: Bundle, tau: float, knn_k: int = 0, nv: int = 0):
    """Eq. (3)-(4): |Pearson r| affinity over TRAIN samples, then thresholded.

    `knn_k > 0` replaces the threshold with a k-nearest-neighbour graph so that
    no node -- in particular neither forecast target -- is left isolated."""
    tr = b.idx("train")
    # use the last frame of every training window -> one row per station-day.
    # With neighbour blocks appended, only the centre station's variables define
    # the variable graph; the blocks reuse it.
    Z = b.X[tr][:, -1, :, 0]
    if nv and Z.shape[1] > nv:
        Z = Z[:, :nv]
    Z = Z - Z.mean(0, keepdims=True)
    sd = Z.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Z = Z / sd
    R = (Z.T @ Z) / len(Z)
    W = np.abs(np.nan_to_num(R))
    np.fill_diagonal(W, 0.0)
    W = W.astype(np.float32)
    if knn_k and knn_k > 0:
        A = knn_graph(W, int(knn_k))
    else:
        A = (W >= tau).astype(np.float32)
        np.fill_diagonal(A, 0.0)
        iso = [NODE_VARS[i] for i in range(A.shape[0]) if A[i].sum() == 0]
        if iso:
            print(f"[warn] tau={tau} leaves {len(iso)} isolated node(s): "
                  f"{', '.join(iso)}")
            tgt = [v for v in iso if v in ("chl", "redtide")]
            if tgt:
                print(f"[warn] {', '.join(tgt)} is a FORECAST TARGET with no edges - "
                      f"the graph pathway cannot reach it.  Use --knn 2.")
    return W, A


def k_adjacency(A: np.ndarray, k: int) -> np.ndarray:
    """Exact k-hop adjacency (Eq. 6): [A^(k)]_ij = 1 iff shortest path == k."""
    N = A.shape[0]
    if k == 0:
        return np.eye(N, dtype=np.float32)
    reach = np.eye(N, dtype=bool)
    cur = np.eye(N, dtype=bool)
    Ab = A.astype(bool)
    for _ in range(k):
        cur = (cur @ Ab) > 0
        prev_reach = reach.copy()
        reach = reach | cur
    exact = cur & ~prev_reach
    return exact.astype(np.float32)


def normalize_adj(A: np.ndarray) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2."""
    At = A + np.eye(A.shape[0], dtype=A.dtype)
    d = At.sum(1)
    d[d == 0] = 1.0
    Dm = np.diag(d ** -0.5)
    return (Dm @ At @ Dm).astype(np.float32)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", default="outputs/daily_station_day.csv")
    ap.add_argument("--out", default="outputs/bundle.npz")
    ap.add_argument("--impute", default="none", choices=["none", "ffill"])
    ap.add_argument("--mask_channel", action="store_true")
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--station_knn", type=int, default=0,
                    help="append K neighbour STATIONS as extra node blocks. The "
                         "inter-station red-tide correlation reaches 0.9-1.0 while "
                         "the variable graph tops out at 0.098, so this is where "
                         "the signal is.")
    ap.add_argument("--log_target", action="store_true",
                    help="fit log1p(Chl-a); reported metrics stay in ug/L")
    ap.add_argument("--T", type=int, default=5, help="input window length (days)")
    ap.add_argument("--knn", type=int, default=0,
                    help="build a k-NN variable graph instead of thresholding "
                         "|r| >= tau.  k=2 guarantees every node -- including "
                         "chl and redtide -- has edges.")
    ap.add_argument("--max_ffill", type=int, default=3)
    ap.add_argument("--variables", default=",".join(WQ_VARS),
                    help="comma-separated node set; 'chl' must be present")
    ap.add_argument("--stations", default="", help="optional comma-separated subset")
    ap.add_argument("--qc", default="none", choices=["none", "physical"],
                    help="physical: blank pH outside 5-10, DO<=0, EC<=0 (+salinity), Chl-a<0 "
                         "before standardisation (revision, R2-m1)")
    ap.add_argument("--extra_feats", default="",
                    help="comma-separated extra input channels broadcast to every node: "
                         "doy (day-of-year sin/cos), rthist (days since last red tide, "
                         "30-day positive count)")
    a = ap.parse_args()

    cfg = DataConfig(daily_path=a.daily, impute=a.impute, max_ffill=a.max_ffill,
                     add_mask_channel=a.mask_channel, corr_threshold=a.tau,
                     knn_k=a.knn, station_knn=a.station_knn,
                     log_target=a.log_target, T=a.T,
                     variables=tuple(a.variables.split(",")),
                     stations=tuple(s for s in a.stations.split(",") if s), qc=a.qc,
                     extra_feats=tuple(f for f in a.extra_feats.split(",") if f))
    b = build(cfg)
    b.save(a.out)

    print(f"samples total : {len(b.X)}   X={b.X.shape}")
    for sp in ["train", "val", "test"]:
        i = b.idx(sp)
        print(f"  {sp:5s} n={len(i):6d}  pos(D+1)={int(b.y_cls[i,0][b.m_cls[i,0]].sum()):4d}"
              f"  chl-valid(D+1)={int(b.m_reg[i,0].sum()):5d}"
              f"  stations={len(np.unique(b.station[i]))}")
    te = b.idx("test")
    ea = te[b.event_adj[te] & b.m_cls[te, 0]]
    print(f"event-adjacent test subset (legacy, D+1-anchored): n={len(ea)}  positives={int(b.y_cls[ea,0].sum())}")
    for hi in range(b.y_cls.shape[1]):
        m = b.event_adj_h[te, hi] & b.m_cls[te, hi]
        print(f"  target-anchored D+{hi+1}: n={int(m.sum()):4d}  positives={int(b.y_cls[te][m, hi].sum()):3d}")
    print("\nvariable graph (|r| >= tau):")
    K = int(b.cfg.get("station_knn", 0))
    lbl = ([f"{v}" for v in NODE_VARS] +
           [f"nb{j}:{v}" for j in range(1, K + 1) for v in NODE_VARS])[:b.A.shape[0]]
    print(pd.DataFrame(b.A, index=lbl, columns=lbl).astype(int).to_string())
    deg = b.A.sum(1)
    iso = [lbl[i] for i in range(len(lbl)) if deg[i] == 0]
    print(f"\nnodes {b.A.shape[0]}, edges {int(b.A.sum() / 2)}, isolated {len(iso)}"
          + (f" -> {', '.join(iso)}" if iso else ""))
    print(f"\nsaved -> {a.out}")
