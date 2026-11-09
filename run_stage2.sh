#!/usr/bin/env bash
# =====================================================================================
#  Stage 2 of the revision: re-run the WHOLE comparison matrix (13 models + 2 single-task
#  controls, 10 seeds, 142 jobs) on each input-channel bundle from stage 1, then run the
#  post-hoc analysis per bundle.  Output tree mirrors outputs/ so analysis/common.py
#  works unchanged with MTHAB_OUT=outputs/stage2_<feats>:
#
#     outputs/stage2_<feats>/run_standard/{runs,results.csv,results_ext.csv}
#     outputs/stage2_<feats>/analysis/*.csv|png
#     outputs/stage2_<feats>/{daily_station_day.csv,ablation}  -> symlinks to outputs/
#
#     bash run_stage2.sh                      # doy rthist doy-rthist (sequential, ~2 h each)
#     FEATS="doy-rthist" bash run_stage2.sh   # one bundle
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
SEEDS=${SEEDS:-10}; EPOCHS=${EPOCHS:-60}; PATIENCE=${PATIENCE:-15}

for f in ${FEATS:-doy rthist doy-rthist}; do
  B="outputs/bundle_ab_k2_s0_l1_T10_qc_f${f}.npz"
  D="outputs/stage2_${f}"
  [ -f "$B" ] || { echo "missing bundle $B"; exit 1; }
  mkdir -p "$D/run_standard" "$D/analysis"
  [ -e "$D/daily_station_day.csv" ] || ln -s ../daily_station_day.csv "$D/daily_station_day.csv"
  [ -e "$D/ablation" ] || ln -s ../ablation "$D/ablation"

  echo "== [$f] stage 2: full matrix on $B   ($(date '+%F %T'))"
  python -m src.run_parallel --bundle "$B" --seeds $SEEDS --epochs $EPOCHS --patience $PATIENCE \
         --station_emb 8 --auto \
         --outdir "$D/run_standard/runs" --results "$D/run_standard/results.csv" \
         --results_ext "$D/run_standard/results_ext.csv"

  echo "== [$f] stage 4: post-hoc analysis   ($(date '+%F %T'))"
  export MTHAB_OUT="$D" MTHAB_FINAL_DIR="$D/run_standard/runs" MTHAB_FINAL_FMT='A_mthab_mse_s{}' \
         MTHAB_BUNDLE="$B" MTHAB_ANALYSIS_OUT="$D/analysis"
  for s in analysis/01_reanchor_eventadj.py analysis/02_persistence_fullyear_events.py \
           analysis/03_onset_station_dataflow.py analysis/04_figures_table6.py \
           analysis/05_threshold_rules.py analysis/06_interpretation.py analysis/07_multitask_vs_single.py \
           analysis/09_significance.py; do
    echo "--- $s" | tee -a "$D/analysis/log.txt"
    python "$s" 2>&1 | tee -a "$D/analysis/log.txt" || echo "!! $s failed" | tee -a "$D/analysis/log.txt"
  done
  python -m src.robust_summary --results "$D/run_standard/results.csv" --out "$D/run_standard/robust" || true
  echo "== [$f] done   ($(date '+%F %T'))"
done
echo "== stage 2 finished for: ${FEATS:-doy rthist doy-rthist}"
