"""
fig_reliability.py — 시드 강건성 그림 (논문용, 흑백 인쇄 안전)

  (a) 시드별 D+1..D+7 평균 RMSE      — 중앙값 + 개별 시드
  (b) 시드별 event-adjacent 평균 F1  — 붕괴(0 근처) 시드가 보이게

색을 쓰지 않습니다. 제안 모델만 검은 채움, 나머지는 회색 빈 표식이라
흑백 인쇄·복사·색약 조건에서 동일하게 읽힙니다.
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

H = [f"D+{h}" for h in range(1, 8)]
LABEL = {"climatology":"Climatology","histgb":"GBT","mlp":"MLP","lstm":"LSTM",
         "gru":"GRU","cnn_lstm":"CNN-LSTM","tcn":"TCN","transformer":"Transformer",
         "stgcn":"ST-GCN","mtgnn":"MTGNN","am_mtgnn":"AM-MTGNN",
         "mtgnn_nograph":"MTGNN (no graph)","mthab_mse":"MT-HAB-G3D",
         "persistence":"Persistence","mthab_cls_only":"MT-HAB (cls only)","mthab_reg_only":"MT-HAB (reg only)"}
PROP = "mthab_mse"
INK, MUTED, GRID = "#111111", "#5A5A5A", "#CFCFCF"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
})


def hcols(d, metric, sub):
    return [f"{sub}|{h}|{metric}" for h in H if f"{sub}|{h}|{metric}" in d.columns]


def panel(ax, per, order, xlabel, collapse_line=None, xmax=None):
    ys = np.arange(len(order))[::-1]
    for y, name in zip(ys, order):
        v = np.asarray(per[name], dtype=float)
        v = v[~np.isnan(v)]
        if not len(v):
            continue
        prop = name == PROP
        med = float(np.median(v))
        # 축 범위를 넘는 값은 경계에 삼각형으로 표시 (축이 이상치에 끌려가지 않게)
        if xmax is not None:
            off = v[v > xmax]
            v_in = v[v <= xmax]
            if len(off):
                ax.plot(np.full(len(off), xmax), np.full(len(off), y),
                        marker=">", linestyle="none", ms=4.0, clip_on=False,
                        mfc="none", mec=MUTED, mew=0.8, zorder=5)
                ax.annotate(", ".join(f"{xv:.1f}" for xv in np.sort(off)),
                            xy=(xmax, y), xytext=(-7, 3.5),
                            textcoords="offset points", fontsize=6,
                            color=MUTED, ha="right", va="bottom")
        else:
            v_in = v
        # 개별 시드
        ax.plot(v_in, np.full(len(v_in), y), marker="o", linestyle="none",
                ms=3.4, mfc=(INK if prop else "none"),
                mec=(INK if prop else MUTED), mew=0.7, alpha=0.95, zorder=3)
        # 중앙값
        med = min(med, xmax) if xmax is not None else med
        ax.plot([med], [y], marker="|", ms=11,
                color=(INK if prop else MUTED),
                mew=(1.6 if prop else 1.0), zorder=4)
        # 범위 선
        if len(v) > 1:
            hi = min(v.max(), xmax) if xmax is not None else v.max()
            ax.plot([v.min(), hi], [y, y], "-",
                    color=(INK if prop else GRID),
                    lw=(1.1 if prop else 0.8), zorder=2)
    if collapse_line is not None:
        ax.axvline(collapse_line, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL.get(n, n) + ("  ◀" if n == PROP else "")
                        for n in order])
    for t, n in zip(ax.get_yticklabels(), order):
        if n == PROP:
            t.set_fontweight("bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(order) - 0.3)
    for s in ax.spines.values():
        s.set_color(INK)
    if xmax is not None:
        ax.set_xlim(right=xmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/robust")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fr = [pd.read_csv(p) for p in a.results
          if os.path.exists(p) and os.path.getsize(p) > 0]
    if not fr:
        raise SystemExit("ERROR: 읽을 수 있는 results CSV 가 없습니다")
    d = pd.concat(fr, ignore_index=True)
    rmse = d[hcols(d, "RMSE", "full_year")].mean(axis=1)
    f1 = d[hcols(d, "F1", "event_adjacent")].mean(axis=1)
    per_r = {n: g.tolist() for n, g in rmse.groupby(d.name)}
    per_f = {n: g.tolist() for n, g in f1.groupby(d.name)}
    order = sorted(per_r, key=lambda n: np.median(per_r[n]))          # 좋은 것부터

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             gridspec_kw=dict(wspace=0.42))
    panel(axes[0], per_r, order, "Chl-a RMSE, mean over D+1–D+7 (μg L$^{-1}$)",
          xmax=4.7)
    axes[0].set_title("(a) forecast error", loc="left", fontweight="bold")

    order_f = sorted(per_f, key=lambda n: -np.median(per_f[n]))
    panel(axes[1], per_f, order_f,
          "Red-tide $F_1$, event-adjacent, mean over D+1–D+7",
          collapse_line=0.02)
    axes[1].set_title("(b) event-detection reliability", loc="left", fontweight="bold")
    axes[1].annotate("task lost\n($F_1\\!\\approx\\!0$)", xy=(0.02, len(order_f) - 1.15),
                     xytext=(0.048, len(order_f) - 1.6), textcoords="data",
                     fontsize=6.5, color=MUTED, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))

    cap = ("Markers: individual seeds; long vertical tick: median; filled black = proposed "
           "model. Open triangles in (a) are seeds beyond the axis, labelled with their "
           "value. Panel (b): most baselines lose the rare-event task entirely on at least "
           "one seed; the proposed model does not.")
    fig.text(0.008, -0.055, cap, fontsize=6.5, color=MUTED, va="top", wrap=True)
    if a.title:
        fig.suptitle(a.title, fontsize=8, y=1.02)

    p = os.path.join(a.out, "fig_reliability.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("->", p)


if __name__ == "__main__":
    main()
