"""Figures for R1-G1 (obs vs pred), R2-m4 / R1-G2 (correlation, k-NN adjacency,
learned adjacency), PR / POD-FAR curves (R2-M1); Table 6 row (R2-m7);
continuation-vs-onset decomposition of true positives."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option("display.width", 250)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

b = load_bundle(BUNDLE_9); stations = b["stations"]
EN = {"광양망덕": "Gwangyang-Mangdeok", "광양적량": "Gwangyang-Jeokryang", "광양초남": "Gwangyang-Chonam", "금강하구": "Geum estuary",
      "낙동명지": "Nakdong-Myeongji", "낙동을숙": "Nakdong-Eulsuk", "마산봉암": "Masan-Bongam", "마산삼귀": "Masan-Samgwi", "마산양덕": "Masan-Yangdeok",
      "부산수영": "Busan-Suyeong", "새만금": "Saemangeum", "시화반월": "Sihwa-Banwol", "시화조력": "Sihwa-tidal", "영산목포": "Yeongsan-Mokpo",
      "영산영암": "Yeongsan-Yeongam", "울산온산": "Ulsan-Onsan", "인천송도": "Incheon-Songdo"}
te = np.where(b["split"] == "test")[0]
P = load_all(["MT-HAB-G3D (proposed)", "Day-of-year climatology", "MLP", "Gradient-boosted trees"])
assert "MT-HAB-G3D (proposed)" in P, "final-model runs not found (set MTHAB_FINAL_DIR / MTHAB_FINAL_FMT)"
mt = P["MT-HAB-G3D (proposed)"]; ref = mt[0]
ev_h = np.load(f"{OUT}/event_adj_per_horizon.npy")
df = load_daily(); d20 = df[df.year == 2020]

# seed-median prediction (per sample) for plotting
prob_med = np.median(np.stack([p["prob"] for p in mt]), 0)
reg_med = np.median(np.stack([p["reg_ugl"] for p in mt]), 0)
thr_med = np.median(np.stack([p["thr"] for p in mt]), 0)

# ------------------------------------------------------------ Fig A: observed vs predicted time series, 3 stations
pick = ["광양적량", "낙동명지", "마산삼귀"]
fig, axes = plt.subplots(len(pick), 2, figsize=(11, 7.2), gridspec_kw=dict(width_ratios=[3, 1.1]))
for r, s in enumerate(pick):
    si = stations.index(s); m = ref["station"] == si
    dd = d20[d20.station == s].sort_values("date")
    ax = axes[r, 0]
    ax.plot(dd.date, dd.chl, color="0.25", lw=0.9, label="observed Chl-a")
    for h, c, ls in [(0, "tab:blue", "-"), (6, "tab:red", "--")]:
        tgt = ref["date"][m] + np.timedelta64(h, "D"); ok = ref["m_reg"][m, h]
        ax.plot(tgt[ok], reg_med[m, h][ok], color=c, lw=0.9, ls=ls, label=f"MT-HAB-G3D D+{h+1}")
    pos = dd[dd.redtide_30km == 1].date
    for d in pos: ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="orange", alpha=0.35, lw=0)
    ax.set_ylabel("Chl-a (µg L$^{-1}$)"); ax.set_title(f"({'abc'[r]}) {EN[s]} — 2020 test year", loc="left", fontsize=9)
    ax2 = ax.twinx(); tgt1 = ref["date"][m]
    ax2.plot(tgt1, prob_med[m, 0], color="tab:green", lw=0.7, alpha=0.8, label="P(red tide) D+1")
    ax2.axhline(thr_med[0], color="tab:green", lw=0.6, ls=":"); ax2.set_ylim(0, 1); ax2.set_ylabel("P(red tide)", color="tab:green")
    ax2.spines["right"].set_visible(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    if r == 0:
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2 + [plt.Rectangle((0, 0), 1, 1, color="orange", alpha=0.35)], l1 + l2 + ["reported red tide (30 km)"], ncol=3, fontsize=7, loc="upper left")
    axs = axes[r, 1]
    for h, c in [(0, "tab:blue"), (6, "tab:red")]:
        ok = ref["m_reg"][m, h]; axs.scatter(ref["y_reg_raw"][m, h][ok], reg_med[m, h][ok], s=6, alpha=0.5, color=c, label=f"D+{h+1}")
    lim = max(1, np.nanmax(dd.chl) * 1.05); axs.plot([0, lim], [0, lim], "k:", lw=0.7); axs.set_xlim(0, lim); axs.set_ylim(0, lim)
    axs.set_xlabel("observed"); axs.set_ylabel("predicted"); axs.set_aspect("equal")
    if r == 0: axs.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/figA_obs_vs_pred_timeseries.png", dpi=170); fig.savefig(f"{OUT}/figA_obs_vs_pred_timeseries.pdf")

# ------------------------------------------------------------ Fig B: pooled scatter per horizon, hexbin + 1:1, with peak-decile underprediction
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, h in zip(axes, [0, 2, 4, 6]):
    ok = ref["m_reg"][:, h]; y = ref["y_reg_raw"][ok, h]; yh = reg_med[ok, h]
    ax.scatter(y, yh, s=4, alpha=0.35, color="tab:blue"); ax.plot([0, 60], [0, 60], "k:", lw=0.8)
    q = np.quantile(y, 0.9); ax.axvline(q, color="tab:red", lw=0.6, ls="--")
    r = reg_metrics(y, yh); ax.set_title(f"D+{h+1}: RMSE {r['RMSE']:.2f}, R² {r['R2']:.2f}, Peak-RMSE {r['PeakRMSE']:.1f}", fontsize=8)
    ax.set_xlim(0, 60); ax.set_ylim(0, 60); ax.set_xlabel("observed Chl-a (µg/L)"); ax.set_aspect("equal")
    ax.text(q + 1, 55, "top decile →", color="tab:red", fontsize=7)
axes[0].set_ylabel("predicted (seed-median)")
fig.tight_layout(); fig.savefig(f"{OUT}/figB_scatter_by_horizon.png", dpi=170); fig.savefig(f"{OUT}/figB_scatter_by_horizon.pdf")

# ------------------------------------------------------------ Fig C: correlation matrix, kNN adjacency, learned adjacency
W = b["W"][:9, :9]; A = b["A"][:9, :9]
# NOTE: metrics.json["learned_adjacency"] is the model's *input* graph (identical to A for
# the fixed-graph configuration), so it is not drawn here; the trainable residual
# adjacency |A_res| is drawn in figF (06_interpretation.py) from learned_params.
LA = None
lab = ["EC", "Temp", "pH", "DO", "Turb", "Sal", "Chl-a", "COD", "Red tide"]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
im = axes[0].imshow(W, cmap="Blues", vmin=0, vmax=0.7); axes[0].set_title("(a) |Pearson r|, training period (pooled)", fontsize=8)
for i in range(9):
    for j in range(9):
        if i != j: axes[0].text(j, i, f"{W[i,j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if W[i, j] > 0.4 else "black")
axes[1].imshow(A, cmap="Greys", vmin=0, vmax=1); axes[1].set_title("(b) symmetric 2-NN adjacency A", fontsize=8)
if LA is not None:
    im2 = axes[2].imshow(LA, cmap="viridis"); axes[2].set_title("(c) learned adjacency (mean of 10 seeds)", fontsize=8); plt.colorbar(im2, ax=axes[2], fraction=0.046)
for ax in axes:
    ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.set_xticklabels(lab, rotation=60, fontsize=6.5); ax.set_yticklabels(lab, fontsize=6.5)
plt.colorbar(im, ax=axes[0], fraction=0.046)
fig.tight_layout(); fig.savefig(f"{OUT}/figC_graphs.png", dpi=170); fig.savefig(f"{OUT}/figC_graphs.pdf")
if LA is not None:
    print("learned adjacency (seed mean):"); print(pd.DataFrame(LA, index=lab, columns=lab).round(3).to_string())
    print("learned adjacency seed-to-seed sd (mean over entries): %.4f" % np.mean(np.std([np.array(json.load(open(f'{FINAL_DIR}/{FINAL_FMT.format(s)}/metrics.json'))['learned_adjacency']) for s in range(10)], 0)))

# ------------------------------------------------------------ Fig D: PR curves (event-adjacent, target-anchored) and POD–FAR, D+1 and D+7
from sklearn.metrics import precision_recall_curve
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
for ax, h in zip(axes, [0, 6]):
    for name, c in [(n_, c_) for n_, c_ in [("MT-HAB-G3D (proposed)", "tab:red"), ("MLP", "tab:blue"), ("Day-of-year climatology", "0.4"), ("Gradient-boosted trees", "tab:green")] if n_ in P]:
        probs = np.median(np.stack([q["prob"] for q in P[name]]), 0)
        m = ref["m_cls"][:, h] & ev_h[:, h]; y = ref["y_cls"][m, h]
        pr, rc, _ = precision_recall_curve(y, probs[m, h]); ax.step(rc, pr, where="post", color=c, lw=1, label=f"{name} (AP {auprc(y, probs[m,h]):.2f})")
    ax.axhline(y.mean(), color="k", ls=":", lw=0.7); ax.text(0.6, y.mean() + 0.02, f"prevalence {y.mean():.2f}", fontsize=7)
    ax.set_xlabel("recall (POD)"); ax.set_ylabel("precision (1 − FAR)"); ax.set_title(f"event-adjacent subset, D+{h+1}", fontsize=8); ax.set_ylim(0, 1); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/figD_pr_curves.png", dpi=170); fig.savefig(f"{OUT}/figD_pr_curves.pdf")

# ------------------------------------------------------------ Table 6 row for the final model (original event_adj definition, as in the paper)
print("\n=== Table 6 row, MT-HAB-G3D (final = no_station_graph), paper definition ===")
div = fail = 0; seeds_fail = 0; worst = []
for p in mt:
    f1s = []; r2s = []
    for h in range(H):
        me = p["m_cls"][:, h] & p["event_adj"]; f1s.append(cls_metrics(p["y_cls"][me, h], p["prob"][me, h], p["thr"][h])["F1"])
        mr = p["m_reg"][:, h]; r2s.append(reg_metrics(p["y_reg_raw"][mr, h], p["reg_ugl"][mr, h])["R2"])
    div += int(min(r2s) < -1); nf = sum(f == 0 for f in f1s); fail += nf; seeds_fail += int(nf > 0); worst.append(np.mean(f1s))
print(f"divergent runs {div}/10, detection failures {fail}/70, seeds with >=1 failure {seeds_fail}/10, worst-seed mean F1 {min(worst):.3f}")
# with the re-anchored subset
fail2 = 0; worst2 = []
for p in mt:
    f1s = [cls_metrics(p["y_cls"][p["m_cls"][:, h] & ev_h[:, h], h], p["prob"][p["m_cls"][:, h] & ev_h[:, h], h], p["thr"][h])["F1"] for h in range(H)]
    fail2 += sum(f == 0 for f in f1s); worst2.append(np.mean(f1s))
print(f"re-anchored: detection failures {fail2}/70, worst-seed mean F1 {min(worst2):.3f}")

# ------------------------------------------------------------ continuation vs onset decomposition of TPs (D+1)
print("\n=== D+1 true positives: was the ORIGIN day (t) already positive? (median over seeds) ===")
RT = NODE_VARS.index("redtide"); rt_last = b["X"][te, -1, RT, 0]
for name in [m for m in ["MT-HAB-G3D (proposed)", "MLP", "Gradient-boosted trees", "Day-of-year climatology"] if m in P]:
    v = []
    for p in P[name]:
        m = p["m_cls"][:, 0]; tp = m & (p["prob"][:, 0] >= p["thr"][0]) & (p["y_cls"][:, 0] == 1)
        v.append((int(tp.sum()), int((tp & (rt_last == 1)).sum())))
    v = np.median(np.array(v), 0); print(f"  {name:26s} TP={v[0]:.0f}, of which origin already positive = {v[1]:.0f}")
m = ref["m_cls"][:, 0]; print(f"  positives at D+1 with origin positive: {int(((ref['y_cls'][:,0]==1)&(rt_last==1)&m).sum())} / {int((ref['y_cls'][m,0]==1).sum())}")
print("figures written to", OUT)
