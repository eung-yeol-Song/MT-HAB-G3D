"""Regenerate manuscript Figs. 3-5 from the final runs.
  Fig_3  horizon dependence: Chl-a RMSE, event-adjacent AUPRC and F1 vs lead time
  Fig_4  seed-paired significance grid (12 comparison rows x 8 metrics)
  Fig_5  component / candidate analysis (Table 7) as paired differences
Writes PDF+PNG to $MTHAB_FIG_OUT (default: $MTHAB_ANALYSIS_OUT)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FIG = os.environ.get("MTHAB_FIG_OUT", OUT); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
ev_h = np.load(f"{OUT}/event_adj_per_horizon.npy")
preds = load_all(verbose=False)
b = load_bundle(BUNDLE_9); tei = np.where(b["split"] == "test")[0]
ref = preds["MT-HAB-G3D (proposed)"][0]
HZ = np.arange(1, H + 1)


def by_h(plist, metric, ev=ev_h):
    out = []
    for p in plist:
        ev = ev if ev.shape[0] == len(p["m_cls"]) else p["event_adj_h"]
        v = []
        for h in range(H):
            if metric in ("RMSE", "R2", "PeakRMSE", "TA1"):
                mr = p["m_reg"][:, h]; v.append(reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h])[metric])
            else:
                m = p["m_cls"][:, h] & ev[:, h]; v.append(cls_metrics(p["y_cls"][m, h], p["prob"][m, h], p["thr"][h])[metric])
        out.append(v)
    return np.array(out)


# persistence rows (Chl-a: last observation; red tide: last label), as in 02_persistence_fullyear_events.py
CHL, RT = NODE_VARS.index("chl"), NODE_VARS.index("redtide")
mu = np.array([b["scaler"][s]["mu"]["chl"] for s in b["stations"]]); sd = np.array([b["scaler"][s]["sd"]["chl"] for s in b["stations"]])
chl_last = b["X"][tei, -1, CHL, 0] * sd[b["station"][tei]] + mu[b["station"][tei]]; ok_last = b["X"][tei, -1, CHL, 1] > 0.5
pers = dict(prob=np.repeat(b["X"][tei, -1, RT, 0][:, None], H, 1), reg_ugl=np.repeat(chl_last[:, None], H, 1),
            y_cls=ref["y_cls"], m_cls=ref["m_cls"], y_reg_raw=ref["y_reg_raw"], m_reg=ref["m_reg"] & ok_last[:, None], thr=np.full(H, 0.5))

# ---------------------------------------------------------------- Fig 3
SHOW = [("MT-HAB-G3D (proposed)", "tab:red", "-", "o"), ("Gradient-boosted trees", "tab:green", "-", "s"), ("MLP", "tab:blue", "-", "^"),
        ("LSTM", "tab:purple", "--", "v"), ("Day-of-year climatology", "0.45", ":", "d")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
panels = [("RMSE", "(a) Chl-$a$ RMSE, full-year timeline (µg L$^{-1}$)"), ("AUPRC", "(b) red-tide AUPRC, event-adjacent subset"), ("F1", "(c) red-tide F1, event-adjacent subset")]
gbt = {m: by_h(preds["Gradient-boosted trees"], m) for m, _ in panels}
for ax, (metric, title) in zip(axes, panels):
    for name, c, ls, mk in SHOW:
        if name not in preds: continue
        v = by_h(preds[name], metric); med = np.median(v, 0)
        ax.plot(HZ, med, color=c, ls=ls, marker=mk, ms=3.5, lw=1.1, label=name)
        if name.startswith("MT-HAB"):
            ax.fill_between(HZ, np.quantile(v, .25, 0), np.quantile(v, .75, 0), color=c, alpha=0.15, lw=0)
            # significance vs GBT per horizon (two-sided paired Wilcoxon over seeds); marker above the curve
            for h in range(H):
                d = v[:, h] - gbt[metric][:, h]
                if np.allclose(d, 0): continue
                p = wilcoxon(v[:, h], gbt[metric][:, h]).pvalue
                if p < 0.05:
                    better = (med[h] < np.median(gbt[metric][:, h])) if metric == "RMSE" else (med[h] > np.median(gbt[metric][:, h]))
                    ax.annotate("*" if better else "°", (HZ[h], med[h]), textcoords="offset points", xytext=(0, 6 if metric != "RMSE" else -11),
                                ha="center", fontsize=9, color="tab:red" if better else "tab:green")
    pv = by_h([pers], metric)[0]
    ax.plot(HZ, pv, color="k", ls="-.", marker="x", ms=3.5, lw=0.9, label="Persistence")
    ax.set_title(title, fontsize=8, loc="left"); ax.set_xlabel("lead time (days)"); ax.set_xticks(HZ); ax.set_xticklabels([f"D+{h}" for h in HZ])
axes[0].legend(fontsize=6.5, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/Fig_3.pdf"); fig.savefig(f"{FIG}/Fig_3.png", dpi=200)

# ---------------------------------------------------------------- Fig 4 (grid from table5_pairs.csv)
t5 = pd.read_csv(f"{OUT}/table5_pairs.csv")
ROWS = ["Gradient-boosted trees", "MLP", "LSTM", "GRU", "CNN-LSTM", "Dilated TCN", "Transformer", "ST-GCN (no G3D)", "MTGNN", "AM-MTGNN", "MTGNN (no graph)", "MT-HAB-G3D + spatial ext."]
METS = ["RMSE", "PeakRMSE", "R2", "F1", "AUPRC", "MCC", "TSS", "acc"]
LABM = ["RMSE", "Peak-RMSE", "$R^2$", "F1", "AUPRC", "MCC", "TSS", "Accuracy"]
fig, ax = plt.subplots(figsize=(4.6, 4.6))
for i, r in enumerate(ROWS):
    for j, m in enumerate(METS):
        g = t5[(t5.comparison == r) & (t5.metric == m)]
        if g.empty: continue
        g = g.iloc[0]; sig = g.p < 0.05; bh = g.q_BH < 0.05
        if sig and g.favours_proposed:
            fc = "tab:red" if bh else "lightcoral"; hatch = None
        elif sig:
            fc = "white"; hatch = "////"
        else:
            fc = "white"; hatch = None
        ax.add_patch(Rectangle((j, len(ROWS) - 1 - i), 1, 1, facecolor=fc, edgecolor="0.3", lw=0.6, hatch=hatch))
ax.set_xlim(0, len(METS)); ax.set_ylim(0, len(ROWS)); ax.set_aspect("equal")
ax.set_xticks(np.arange(len(METS)) + 0.5); ax.set_xticklabels(LABM, rotation=60, ha="right", fontsize=7)
ax.set_yticks(np.arange(len(ROWS)) + 0.5); ax.set_yticklabels([r.replace("MT-HAB-G3D + spatial ext.", "+ spatial extension") for r in ROWS[::-1]], fontsize=7)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
hs = [Rectangle((0, 0), 1, 1, facecolor="tab:red", edgecolor="0.3"), Rectangle((0, 0), 1, 1, facecolor="lightcoral", edgecolor="0.3"),
      Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.3"), Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.3", hatch="////")]
ax.legend(hs, ["MT-HAB-G3D better, $p<0.05$ and BH $q<0.05$", "MT-HAB-G3D better, raw $p<0.05$ only", "no significant difference", "comparison better, $p<0.05$"],
          fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1)
cnt = pd.read_csv(f"{OUT}/table5_counts.csv").iloc[0]
ax.set_title(f"{int(cnt.favour_p05)} of {int(cnt.n_comparisons)} comparisons favour MT-HAB-G3D ({int(cnt.favour_BH)} after BH); {int(cnt.against_p05)} against", fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{FIG}/Fig_4.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/Fig_4.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------- Fig 5 (ablation / candidates, paired differences from the final model)
ABLROWS = [("plus_station_graph", "+ spatial extension"), ("no_station_emb", "$-$ station embedding"), ("window_5", "$T=5$ window$^{\\ddagger}$"),
           ("lam_0.2", "$\\lambda=0.2$"), ("lam_2.0", "$\\lambda=2.0$"), ("residual_head", "persistence-anchored head"), ("wbce", "class-weighted BCE"),
           ("plus_doy", "+ day-of-year"), ("plus_rthist", "+ red-tide history"), ("plus_doy_rthist", "+ day-of-year & history")]
refv = {m: by_h(preds["MT-HAB-G3D (proposed)"], m).mean(1) for m in ["RMSE", "AUPRC", "F1"]}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, (metric, lab) in zip(axes, [("RMSE", "$\\Delta$ Chl-$a$ RMSE (µg L$^{-1}$)"), ("AUPRC", "$\\Delta$ event-adjacent AUPRC"), ("F1", "$\\Delta$ event-adjacent F1")]):
    for i, (cfg, name) in enumerate(ABLROWS):
        pl = [load_pred(ABL, f"{cfg}_s{s}") for s in range(10) if os.path.exists(f"{ABL}/{cfg}_s{s}/preds_test.npz")]
        if not pl: continue
        v = by_h(pl, metric).mean(1); d = v[:len(refv[metric])] - refv[metric][:len(v)]
        p = wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0
        y = len(ABLROWS) - 1 - i
        ax.scatter(d, np.full(len(d), y) + np.random.default_rng(0).uniform(-0.18, 0.18, len(d)), s=9, color="0.6", alpha=0.7, lw=0)
        ax.plot([np.quantile(d, .25), np.quantile(d, .75)], [y, y], color="k", lw=2)
        ax.scatter([np.median(d)], [y], s=40, facecolor="white" if cfg == "window_5" else ("tab:red" if p < 0.05 else "k"), edgecolor="k", zorder=3)
        ax.text(ax.get_xlim()[1] if False else 0, y + 0.32, f"{np.median(d):+.3f}  (p={p:.3f})", fontsize=6, ha="center", color="tab:red" if p < 0.05 else "0.25")
    ax.axvline(0, color="k", ls="--", lw=0.8); ax.set_xlabel(lab); ax.set_ylim(-0.6, len(ABLROWS) - 0.2)
axes[0].set_yticks(range(len(ABLROWS))); axes[0].set_yticklabels([n for _, n in ABLROWS][::-1], fontsize=7)
fig.suptitle("Seed-paired difference from the final configuration (10 seeds; dots = seeds, bar = IQR, marker = median; red = $p<0.05$)", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/Fig_5.pdf"); fig.savefig(f"{FIG}/Fig_5.png", dpi=200)
print("written", FIG)
