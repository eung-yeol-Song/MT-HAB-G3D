#!/usr/bin/env bash
# =====================================================================================
#  T=5 protocol benchmark: re-run the WHOLE comparison matrix (13 models + 2 single-task
#  controls, 10 seeds, 142 jobs) on the 5-day-window bundle, GPU workers only (the same
#  device policy as the final T=10 reference, see run_regpu.sh), then the post-hoc
#  analysis.  Output tree mirrors outputs/ so analysis/common.py works unchanged with
#  MTHAB_OUT=outputs/t5_full:
#
#     outputs/t5_full/run_standard/{runs,results.csv,results_ext.csv}
#     outputs/t5_full/analysis/*.csv|png
#     outputs/t5_full/{daily_station_day.csv,ablation}  -> symlinks to outputs/
#
#     nohup bash run_t5.sh > t5.log 2>&1 &
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
SEEDS=${SEEDS:-10}; EPOCHS=${EPOCHS:-60}; PATIENCE=${PATIENCE:-15}; GPU_WORKERS=${GPU_WORKERS:-6}

B="outputs/bundle_ab_k2_s0_l1_T5_qc.npz"
D="outputs/t5_full"
[ -f "$B" ] || { echo "missing bundle $B"; exit 1; }
mkdir -p "$D/run_standard" "$D/analysis"
[ -e "$D/daily_station_day.csv" ] || ln -s ../daily_station_day.csv "$D/daily_station_day.csv"
[ -e "$D/ablation" ] || ln -s ../ablation "$D/ablation"

echo "== [T5] full matrix on $B, GPU workers=$GPU_WORKERS   ($(date '+%F %T'))"
python -m src.run_parallel --bundle "$B" --seeds $SEEDS --epochs $EPOCHS --patience $PATIENCE \
       --station_emb 8 --gpu_workers "$GPU_WORKERS" --cpu_workers 0 \
       --outdir "$D/run_standard/runs" --results "$D/run_standard/results.csv" \
       --results_ext "$D/run_standard/results_ext.csv"

echo "== [T5] post-hoc analysis   ($(date '+%F %T'))"
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
echo "== [T5] done   ($(date '+%F %T')) ; failures: $(grep -c '!! ' "$D/analysis/log.txt" || true)"
