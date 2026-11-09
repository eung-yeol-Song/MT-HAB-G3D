"""
robust_summary.py  —  results.csv 를 시드-강건 요약으로 바꿉니다.

왜 필요한가
-----------
이 데이터셋(학습 양성 81, 검증 양성 54)에서는 두 종류의 드문 실패가 관찰됩니다.

  1) 회귀 발산   : 시드 10개 중 1~2개가 장기 선행시간에서 R2 << 0 으로 폭발.
                   평균 RMSE 가 이 시드 하나에 지배됩니다
                   (예: MT-HAB seed 6 → 중앙값 3.56 vs 평균 4.08).
  2) 분류 붕괴   : 어떤 시드에서 임계값이 무너져 F1 = 0 이 됩니다.
                   평균 F1 은 "붕괴 빈도"를 재는 지표가 되어버립니다.

평균 하나로 요약하면 두 실패가 모두 성능 차이로 오독됩니다. 이 스크립트는
전 모델에 **동일한 규칙**을 적용해 평균·중앙값·발산수·붕괴수를 같이 냅니다.
특정 모델에만 유리한 필터가 아니므로 심사에서 방어 가능합니다.

사용법
------
    python -m src.robust_summary --results outputs/run_standard/results.csv \
                                 --out outputs/run_standard/robust
    # 여러 실행을 합치기 (예: 3시드 실행 + 10시드 실행)
    python -m src.robust_summary --results a.csv b.csv --out outputs/robust_pooled

출력
----
    robust_summary.csv   모델별 요약 (평균/중앙값/IQR/발산·붕괴 카운트)
    per_seed_rmse.csv    시드별 D+1..D+7 평균 RMSE 원자료
    per_seed_f1.csv      시드별 event-adjacent F1 원자료
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

H = [f"D+{h}" for h in range(1, 8)]
LABEL = {
    "climatology": "Day-of-year climatology", "histgb": "Gradient-boosted trees",
    "mlp": "MLP", "lstm": "LSTM", "gru": "GRU", "cnn_lstm": "CNN-LSTM",
    "tcn": "Dilated TCN", "transformer": "Transformer", "stgcn": "ST-GCN (no G3D)",
    "mtgnn": "MTGNN", "am_mtgnn": "AM-MTGNN", "mtgnn_nograph": "MTGNN (no graph)",
    "mthab_mse": "MT-HAB-G3D (proposed)",
    "persistence": "Persistence", "mthab_cls_only": "MT-HAB-G3D (classification only)",
    "mthab_reg_only": "MT-HAB-G3D (regression only)",
}
# single-task controls: the untrained head must not be scored (revision)
DEAD_HEAD = {"mthab_cls_only": "reg", "mthab_reg_only": "cls"}
# 사전 선언된 판정 기준 (결과를 보고 정하지 않았음을 명시하기 위해 상수로 둡니다)
DIVERGE_R2 = -1.0     # 어느 선행시간이든 R2 < -1.0 이면 그 시드는 '발산'
COLLAPSE_F1 = 0.02    # event-adjacent F1 이 이 값 미만이면 그 시드는 '분류 붕괴'


def _hcols(df, metric, sub):
    return [f"{sub}|{h}|{metric}" for h in H if f"{sub}|{h}|{metric}" in df.columns]


def load(paths):
    fr = []
    for p in paths:
        # --single_bundle 실행에서는 results_ext.csv 가 없을 수 있습니다.
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"  (건너뜀: {p} 없음)")
            continue
        d = pd.read_csv(p).copy()
        d = pd.concat([d, pd.DataFrame({"source": [os.path.basename(p)] * len(d)},
                                       index=d.index)], axis=1)
        fr.append(d)
    if not fr:
        raise SystemExit("ERROR: 읽을 수 있는 results CSV 가 없습니다")
    d = pd.concat(fr, ignore_index=True).copy()
    # 같은 (name, seed) 가 여러 파일에 있으면 source 로 구분해 별개 실행으로 취급
    d = pd.concat([d, pd.DataFrame(
        {"run_id": d["source"] + "#s" + d["seed"].astype(str)}, index=d.index)],
        axis=1)
    return d


def summarise(d):
    rows = []
    for name, g in d.groupby("name"):
        rmse = g[_hcols(g, "RMSE", "full_year")].mean(axis=1)
        r2h = g[_hcols(g, "R2", "full_year")]
        f1 = g[_hcols(g, "F1", "event_adjacent")].mean(axis=1)
        f1h = g[_hcols(g, "F1", "event_adjacent")]
        aup = g[_hcols(g, "AUPRC", "event_adjacent")].mean(axis=1)

        # single-task controls: the head that received no loss is untrained, so its
        # metrics are not a model result (reg-only "F1 0.45" = predicting all-positive)
        dead = DEAD_HEAD.get(name)
        if dead == "reg":
            rmse = rmse * np.nan; r2h = r2h * np.nan
        elif dead == "cls":
            f1 = f1 * np.nan; f1h = f1h * np.nan; aup = aup * np.nan

        n = len(g)
        diverged = int((r2h.min(axis=1) < DIVERGE_R2).sum()) if len(r2h.columns) else 0
        collapsed = int((f1h.mean(axis=1) < COLLAPSE_F1).sum()) if len(f1h.columns) else 0
        # 선행시간 하나라도 F1=0 이 된 시드
        any_zero = int((f1h.min(axis=1) <= 0.0).sum()) if len(f1h.columns) else 0

        clean = rmse[(r2h.min(axis=1) >= DIVERGE_R2)] if len(r2h.columns) else rmse
        rows.append(dict(
            name=name, Model=LABEL.get(name, name), n_seeds=n,
            RMSE_mean=rmse.mean(), RMSE_median=rmse.median(),
            RMSE_sd=rmse.std(ddof=1) if n > 1 else 0.0,
            RMSE_IQR=(rmse.quantile(.75) - rmse.quantile(.25)) if n > 1 else 0.0,
            RMSE_mean_nodiv=clean.mean() if len(clean) else np.nan,
            F1ea_mean=f1.mean(), F1ea_median=f1.median(),
            F1ea_min=f1.min(), AUPRCea_mean=aup.mean(),
            n_diverged=diverged, n_collapsed=collapsed, n_any_horizon_F1_zero=any_zero,
            diverge_rate=diverged / n, collapse_rate=collapsed / n,
        ))
    s = pd.DataFrame(rows).sort_values("RMSE_median").reset_index(drop=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/robust")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = load(a.results)
    s = summarise(d)

    # 시드 수를 결과 옆에 같이 찍습니다 (3시드를 10시드로 오독하는 사고 방지)
    n_seeds = int(s.n_seeds.max())
    if n_seeds < 10:
        print("\n" + "!" * 68)
        print(f"!! 이 요약은 시드 {n_seeds}개 실행에서 나왔습니다. 논문에 쓰지 마십시오.")
        print("!! 이 데이터셋은 10시드 중 1~2개가 발산하므로, 시드가 적으면")
        print("!! 평균이 '어떤 실패를 우연히 비껴갔는지'를 재게 됩니다.")
        print("!" * 68)
    s.to_csv(os.path.join(a.out, "robust_summary.csv"), index=False)

    rm = pd.DataFrame({"name": d.name, "run": d.run_id,
                       "RMSE": d[_hcols(d, "RMSE", "full_year")].mean(axis=1)})
    rm.pivot_table(index="name", columns="run", values="RMSE").to_csv(
        os.path.join(a.out, "per_seed_rmse.csv"))
    f1 = pd.DataFrame({"name": d.name, "run": d.run_id,
                       "F1": d[_hcols(d, "F1", "event_adjacent")].mean(axis=1)})
    f1.pivot_table(index="name", columns="run", values="F1").to_csv(
        os.path.join(a.out, "per_seed_f1.csv"))

    show = ["Model", "n_seeds", "RMSE_mean", "RMSE_median", "RMSE_sd",
            "F1ea_mean", "F1ea_median", "F1ea_min",
            "n_diverged", "n_collapsed"]
    print(f"\n판정 기준(사전 선언): 발산 = 어느 선행시간이든 R2 < {DIVERGE_R2}, "
          f"붕괴 = event-adjacent F1 < {COLLAPSE_F1}\n")
    print(s[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {a.out}/robust_summary.csv")


if __name__ == "__main__":
    main()
