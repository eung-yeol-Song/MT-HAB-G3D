#!/usr/bin/env bash
# =====================================================================================
#  Refit experiment (revision, 2026-11-10): does the proposed model's position change
#  when every model is refitted on train + validation (2015-2019) before the 2020 test?
#
#  Nothing is tuned here.  All settings are the validation-chosen reference ones.  Because
#  the validation year is used for fitting, early stopping is replaced by a FIXED epoch
#  budget per model = median best validation epoch of its ten reference seeds
#  (src.run_parallel --epochs_from), and decision thresholds are carried over.
#
#    fixed : train only (2015-2018), fixed budget, thresholds fitted on 2019   -> control
#    refit : train + validation (2015-2019), same budget, thresholds of the matching
#            fixed run                                                         -> experiment
#
#  refit - fixed isolates the effect of the extra year; fixed - reference is the cost of
#  dropping checkpoint selection.  Both trees mirror outputs/ so analysis/common.py works
#  with MTHAB_OUT=outputs/refit/<fixed|refit>.
#
#     setsid nohup bash run_refit.sh > refit.log 2>&1 < /dev/null &
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
SEEDS=${SEEDS:-10}; GPU_WORKERS=${GPU_WORKERS:-12}; ONLY=${ONLY:-}
REF="outputs/run_standard/runs"; B="outputs/bundle_main.npz"; R="outputs/refit"

for V in fixed refit; do
  D="$R/$V"
  mkdir -p "$D/run_standard" "$D/analysis"
  [ -e "$D/daily_station_day.csv" ] || ln -s ../../daily_station_day.csv "$D/daily_station_day.csv"
  [ -e "$D/ablation" ] || ln -s ../../ablation "$D/ablation"
  EXTRA=(); [ "$V" = refit ] && EXTRA=(--refit --thr_from "$R/fixed/run_standard/runs")
  [ -n "$ONLY" ] && EXTRA+=(--only "$ONLY")
  # gradient-boosted trees are CPU-bound (OpenMP, all cores): ten of them side by side
  # oversubscribe the machine and stall for hours, so they go first, two at a time
  if [ -z "$ONLY" ]; then
    echo "== [refit:$V] gradient-boosted trees, 2 workers   ($(date '+%F %T'))"
    python -m src.run_parallel --single_bundle --seeds $SEEDS --epochs 60 --patience 15 \
           --station_emb 8 --gpu_workers 2 --cpu_workers 0 --only histgb \
           --outdir "$D/run_standard/runs" --results "$D/run_standard/results_histgb.csv" \
           --results_ext "$D/run_standard/results_ext_histgb.csv" "${EXTRA[@]}"
  fi
  echo "== [refit:$V] matrix, GPU workers=$GPU_WORKERS   ($(date '+%F %T'))"
  python -m src.run_parallel --single_bundle --seeds $SEEDS --epochs 60 --patience 15 \
         --station_emb 8 --gpu_workers "$GPU_WORKERS" --cpu_workers 0 --epochs_from "$REF" \
         --outdir "$D/run_standard/runs" --results "$D/run_standard/results.csv" \
         --results_ext "$D/run_standard/results_ext.csv" "${EXTRA[@]}"
done
[ -n "$ONLY" ] && { echo "== [refit] ONLY=$ONLY given: skipping analysis"; exit 0; }

for V in fixed refit; do
  D="$R/$V"
  echo "== [refit:$V] post-hoc analysis   ($(date '+%F %T'))"
  export MTHAB_OUT="$D" MTHAB_FINAL_DIR="$D/run_standard/runs" MTHAB_FINAL_FMT='A_mthab_mse_s{}' \
         MTHAB_BUNDLE="$B" MTHAB_ANALYSIS_OUT="$D/analysis"
  for s in analysis/01_reanchor_eventadj.py analysis/02_persistence_fullyear_events.py \
           analysis/03_onset_station_dataflow.py analysis/07_multitask_vs_single.py \
           analysis/09_significance.py; do
    echo "--- $s" | tee -a "$D/analysis/log.txt"
    python "$s" 2>&1 | tee -a "$D/analysis/log.txt" || echo "!! $s failed" | tee -a "$D/analysis/log.txt"
  done
done
unset MTHAB_OUT MTHAB_FINAL_DIR MTHAB_FINAL_FMT MTHAB_BUNDLE MTHAB_ANALYSIS_OUT
python analysis/14_refit_compare.py 2>&1 | tee "$R/refit_compare.txt" || echo "!! 14_refit_compare failed"
echo "== [refit] done   ($(date '+%F %T'))"
