#!/usr/bin/env bash
# =====================================================================================
#  MT-HAB-G3D  —  ECOINF revision pipeline (v13-rev1)
#  One command reproduces every number the revised manuscript needs.
#
#     bash run_revision.sh                # GPU box: auto-detects workers
#     bash run_revision.sh --cpu          # CPU only (slow: ~10 h per MT-HAB seed on 2 cores)
#     bash run_revision.sh --smoke        # 1 seed, 2 epochs, 2 models: checks the patched code runs
#
#  Stages (each stage is resumable; finished runs are skipped):
#     0  bundle_main.npz        9 nodes, physical QC, ffill<=3d + mask channel, log1p target, T=10, 2-NN
#     1  reference rows          persistence, climatology, histgb x10        (torch-free, minutes)
#     2  neural matrix           12 neural models x 10 seeds + 2 single-task controls x 10 seeds
#     3  ablation                + spatial ext., - log target, T=5, - station embedding (x10 seeds)
#     4  post-hoc analysis       analysis/01..04 -> tables, figures (numpy only)
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8

MODE="gpu"; SEEDS=10; EPOCHS=60; PATIENCE=15
for a in "$@"; do
  case "$a" in
    --cpu)   MODE="cpu" ;;
    --smoke) MODE="smoke"; SEEDS=1; EPOCHS=2; PATIENCE=1 ;;
  esac
done

echo "== stage 0: bundle"
if [ ! -f outputs/bundle_main.npz ] || ! grep -q "QC=physical" outputs/bundle_main.cfg 2>/dev/null; then
  python -m src.data.dataset --daily outputs/daily_station_day.csv --out outputs/bundle_main.npz \
         --impute ffill --mask_channel --T 10 --knn 2 --log_target --qc physical
  echo "KNN=2 STATION_KNN=0 LOG_TARGET=1 WINDOW=10 QC=physical IMPUTE=ffill3 MASK=1" > outputs/bundle_main.cfg
fi

echo "== stage 1: torch-free references"
python -m src.run_reference --bundle outputs/bundle_main.npz --outdir outputs/run_standard/runs \
       --results outputs/run_standard/results.csv --models persistence,climatology,histgb --seeds $SEEDS

echo "== stage 2: neural matrix"
if [ "$MODE" = "smoke" ]; then
  python -m src.run_parallel --single_bundle --seeds 1 --epochs 2 --patience 1 --station_emb 8 \
         --only lstm,mthab_mse,mthab_cls_only,mthab_reg_only --auto \
         --outdir outputs/smoke/runs --results outputs/smoke/results.csv --results_ext outputs/smoke/results_ext.csv
  python smoke_check.py outputs/smoke/runs
  echo "smoke test finished -> outputs/smoke"; exit 0
elif [ "$MODE" = "cpu" ]; then
  python -m src.run_parallel --single_bundle --seeds $SEEDS --epochs $EPOCHS --patience $PATIENCE \
         --station_emb 8 --gpu_workers 0 --cpu_workers 1 \
         --outdir outputs/run_standard/runs --results outputs/run_standard/results.csv \
         --results_ext outputs/run_standard/results_ext.csv
else
  python -m src.run_parallel --single_bundle --seeds $SEEDS --epochs $EPOCHS --patience $PATIENCE \
         --station_emb 8 --gpu_workers "${GPU_WORKERS:-6}" --cpu_workers 0 \
         --outdir outputs/run_standard/runs --results outputs/run_standard/results.csv \
         --results_ext outputs/run_standard/results_ext.csv
fi

echo "== stage 3: ablation (reference row reused from the main run)"
DEV=cuda; [ "$MODE" = "cpu" ] && DEV=cpu
EXT=""; [ "${EXTENDED:-0}" = "1" ] && EXT="--extended"      # EXTENDED=1 adds lambda sweep + residual head
python -m src.run_ablation --device $DEV --seeds $SEEDS --epochs $EPOCHS --patience $PATIENCE \
       --reuse_main outputs/run_standard/runs --workers "${ABL_WORKERS:-2}" $EXT

# STAGE1=1 : revision candidates of the proposed model (seasonal / red-tide-history channels,
#            class-weighted BCE, residual head, lambda sweep), 10 seeds each.  The winner is
#            chosen on the 2019 validation split by analysis/08_validation_selection.py; if an
#            input-channel row wins, re-run every baseline on that bundle (stage 2):
#              python -m src.run_parallel --bundle outputs/bundle_ab_k2_s0_l1_T10_qc_f<feats>.npz \
#                     --seeds 10 --epochs 60 --patience 15 --station_emb 8 --auto \
#                     --outdir outputs/run_stage2/runs --results outputs/run_stage2/results.csv \
#                     --results_ext outputs/run_stage2/results_ext.csv
#            and point MTHAB_FINAL_DIR / MTHAB_OUT at outputs/run_stage2 for stage 4.
if [ "${STAGE1:-0}" = "1" ]; then
  echo "== stage 3b: validation-selected candidates (proposed model only)"
  python -m src.run_ablation --device $DEV --seeds $SEEDS --epochs $EPOCHS --patience $PATIENCE \
         --reuse_main outputs/run_standard/runs --workers "${ABL_WORKERS:-6}" --stage1
fi

echo "== stage 4: post-hoc analysis"
export MTHAB_OUT=outputs MTHAB_FINAL_DIR="${MTHAB_FINAL_DIR:-outputs/run_standard/runs}" \
       MTHAB_FINAL_FMT="${MTHAB_FINAL_FMT:-A_mthab_mse_s{}}" \
       MTHAB_BUNDLE="${MTHAB_BUNDLE:-outputs/bundle_main.npz}" MTHAB_ANALYSIS_OUT=outputs/analysis
mkdir -p outputs/analysis
for s in analysis/01_reanchor_eventadj.py analysis/02_persistence_fullyear_events.py \
         analysis/03_onset_station_dataflow.py analysis/04_figures_table6.py \
         analysis/05_threshold_rules.py analysis/06_interpretation.py analysis/07_multitask_vs_single.py \
         analysis/08_validation_selection.py analysis/09_significance.py analysis/10_manuscript_tables.py; do
  echo "--- $s" | tee -a outputs/analysis/log.txt
  python "$s" 2>&1 | tee -a outputs/analysis/log.txt
done
python -m src.robust_summary --results outputs/run_standard/results.csv --out outputs/run_standard/robust 2>/dev/null || true
echo "== done. tables: outputs/analysis/*.csv  figures: outputs/analysis/*.png"
