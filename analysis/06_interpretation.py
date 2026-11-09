"""Model interpretation figures (R1-G2, R2-m4) from the artefacts that train.py
writes in the revision pipeline:

  metrics.json["learned_params"]           per-block residual adjacency masks (Eq. 3),
                                           temporal-lag gates (Eq. 7), station embedding
  metrics.json["permutation_importance"]   variable-node importance (dAUPRC, dRMSE) per horizon

Outputs
  figE_permutation_importance.png   nodes x horizons heatmaps (red tide / Chl-a), seed median
  figF_learned_graph.png            prior 2-NN graph vs mean |residual mask| (block 1), gates table
  figG_station_embedding.png        PCA of the learned station embedding, coloured by bloom frequency
  tables: perm_importance.csv, residual_mask_top_edges.csv, temporal_gates.csv

`--demo` fabricates the artefacts (random numbers) so the plotting code can be
checked before the GPU run has finished; demo output is NOT reportable.
"""
import sys, os, json, argparse; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8})

ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true"); a = ap.parse_args()
base, fmt, n = MODELS["MT-HAB-G3D (proposed)"]
LAB = ["EC", "Temp", "pH", "DO", "Turb", "Sal", "Chl-a", "COD", "Red tide"]
b = load_bundle(BUNDLE_9); stations = b["stations"]
A_prior = b["A"][:9, :9]

metrics = []
for s in range(n):
    f = f"{base}/{fmt.format(s)}/metrics.json"
    if os.path.exists(f):
        m = json.load(open(f))
        if "permutation_importance" in m or "learned_params" in m:
            metrics.append(m)
if a.demo or not metrics:
    print("[DEMO] fabricating artefacts - output is NOT reportable")
    rng = np.random.default_rng(0); metrics = []
    for s in range(3):
        lp = {}
        for blk in range(3):
            lp[f"model.blocks.{blk}.gcn.res_mask"] = (rng.normal(0, 0.05, (3, 9, 9))).tolist()
            for tau in (3, 5):
                lp[f"model.blocks.{blk}.g3d_{tau}.res_mask"] = (rng.normal(0, 0.05, (3, 9, 9))).tolist()
                lp[f"model.blocks.{blk}.g3d_{tau}.alpha_t"] = [float(rng.normal(0, 1))]
        lp["emb.weight"] = rng.normal(0, 0.3, (len(stations), 8)).tolist()
        pi = {"per_node": {v: {"dAUPRC": rng.normal(0.01, 0.01, 7).tolist(), "dRMSE": rng.normal(0.05, 0.05, 7).tolist()} for v in NODE_VARS},
              "base_AUPRC": [0.1] * 7, "base_RMSE": [3.0] * 7}
        metrics.append({"learned_params": lp, "permutation_importance": pi})

# ------------------------------------------------------------------ permutation importance
pis = [m["permutation_importance"] for m in metrics if "permutation_importance" in m]
if pis:
    dA = np.median([[p["per_node"][v]["dAUPRC"] for v in NODE_VARS] for p in pis], 0)   # (9, 7)
    dR = np.median([[p["per_node"][v]["dRMSE"] for v in NODE_VARS] for p in pis], 0)
    tab = pd.DataFrame({"node": LAB, "dAUPRC_mean": dA.mean(1), "dRMSE_mean": dR.mean(1)})
    for h in range(H):
        tab[f"dAUPRC_D+{h+1}"] = dA[:, h]; tab[f"dRMSE_D+{h+1}"] = dR[:, h]
    tab.to_csv(f"{OUT}/perm_importance.csv", index=False)
    print("=== permutation importance (median over seeds; mean over horizons) ===")
    print(tab[["node", "dAUPRC_mean", "dRMSE_mean"]].sort_values("dRMSE_mean", ascending=False).round(4).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, M, ttl, cm in [(axes[0], dA, "(a) red tide: loss of full-year AUPRC", "Reds"), (axes[1], dR, "(b) Chl-a: increase of RMSE (µg/L)", "Blues")]:
        im = ax.imshow(M, cmap=cm, aspect="auto", vmin=0); ax.set_yticks(range(9)); ax.set_yticklabels(LAB)
        ax.set_xticks(range(H)); ax.set_xticklabels([f"D+{h+1}" for h in range(H)]); ax.set_title(ttl, fontsize=8)
        for i in range(9):
            for j in range(H):
                ax.text(j, i, f"{M[i,j]:.2f}" if M.max() > 0.5 else f"{M[i,j]:.3f}", ha="center", va="center", fontsize=5.5)
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Permutation importance of input variable nodes (MT-HAB-G3D, seed median)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/figE_permutation_importance.png", dpi=170); fig.savefig(f"{OUT}/figE_permutation_importance.pdf")

# ------------------------------------------------------------------ residual masks and temporal gates
lps = [m["learned_params"] for m in metrics if "learned_params" in m]
if lps:
    def collect(suffix):
        out = {}
        for lp in lps:
            for k, v in lp.items():
                if k.endswith(suffix):
                    out.setdefault(k, []).append(np.array(v))
        return out
    masks = collect("res_mask")
    gates = collect("alpha_t")
    if masks:
        # mean |res_mask| over seeds, scales and all modules (9x9 centre block only)
        agg = np.zeros((9, 9)); cnt = 0
        per_block = {}
        for k, arrs in masks.items():
            A = np.mean([np.abs(x) for x in arrs], 0)             # (K, N, N)
            A9 = A[:, :9, :9].sum(0); agg += A9; cnt += 1
            blk = k.split(".")[2] if ".blocks." in k else "0"
            per_block[blk] = per_block.get(blk, 0) + A9
        agg /= max(cnt, 1)
        np.fill_diagonal(agg, 0)
        edges = [(LAB[i], LAB[j], agg[i, j], int(A_prior[i, j])) for i in range(9) for j in range(9) if i < j]
        te = pd.DataFrame(edges, columns=["node_i", "node_j", "mean_abs_res_mask", "in_prior_graph"]).sort_values("mean_abs_res_mask", ascending=False)
        te.to_csv(f"{OUT}/residual_mask_top_edges.csv", index=False)
        print("\n=== strongest learned residual edges (mean |A_res| over seeds, scales, modules) ===")
        print(te.head(12).round(4).to_string(index=False))
        print(f"share of residual mass on edges absent from the 2-NN prior: "
              f"{te.loc[te.in_prior_graph==0,'mean_abs_res_mask'].sum()/te.mean_abs_res_mask.sum():.2f}")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        axes[0].imshow(A_prior, cmap="Greys", vmin=0, vmax=1); axes[0].set_title("(a) data-derived 2-NN adjacency (fixed)", fontsize=8)
        im = axes[1].imshow(agg, cmap="viridis"); axes[1].set_title("(b) learned residual adjacency |A_res| (seed mean)", fontsize=8)
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        for ax in axes:
            ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.set_xticklabels(LAB, rotation=60, fontsize=6.5); ax.set_yticklabels(LAB, fontsize=6.5)
        fig.tight_layout(); fig.savefig(f"{OUT}/figF_learned_graph.png", dpi=170); fig.savefig(f"{OUT}/figF_learned_graph.pdf")
    if gates:
        g = pd.DataFrame([{"module": k, "gate_sigmoid_mean": float(np.mean([1 / (1 + np.exp(-np.array(x))) for x in arrs])),
                           "gate_sigmoid_sd_seeds": float(np.std([1 / (1 + np.exp(-np.array(x))) for x in arrs]))} for k, arrs in gates.items()])
        g.to_csv(f"{OUT}/temporal_gates.csv", index=False)
        print("\n=== temporal-lag gates gamma_d = sigmoid(alpha_t) (Eq. 7), per G3D module ===")
        print(g.round(3).to_string(index=False))
    embs = [np.array(lp["emb.weight"]) for lp in lps if "emb.weight" in lp]
    if embs:
        E = np.mean(embs, 0)                                           # seeds are not aligned in general; use seed 0 for PCA
        E0 = embs[0]; E0 = E0 - E0.mean(0)
        U_, S_, Vt = np.linalg.svd(E0, full_matrices=False); pc = E0 @ Vt[:2].T
        df = load_daily(); freq = df[df.year.between(2015, 2018)].groupby("station")["redtide_30km"].mean()
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=[freq.get(s, 0) for s in stations], cmap="OrRd", s=40, edgecolor="k", lw=0.4)
        for i, s in enumerate(stations): ax.annotate(EN.get(s, s), (pc[i, 0], pc[i, 1]), fontsize=6, xytext=(3, 2), textcoords="offset points")
        plt.colorbar(sc, label="training-period red-tide frequency")
        ax.set_xlabel(f"PC1 ({S_[0]**2/np.sum(S_**2)*100:.0f}%)"); ax.set_ylabel(f"PC2 ({S_[1]**2/np.sum(S_**2)*100:.0f}%)")
        ax.set_title("Learned station embedding (seed 0), PCA", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{OUT}/figG_station_embedding.png", dpi=170); fig.savefig(f"{OUT}/figG_station_embedding.pdf")
print("done ->", OUT)
