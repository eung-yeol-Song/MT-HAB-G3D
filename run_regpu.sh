#!/usr/bin/env bash
# =====================================================================================
#  Unify the reference matrix on one device.  `run_parallel --auto` had placed 35 of the
#  142 reference jobs on CPU workers (incl. the proposed model's seeds 6-9); same seed on
#  CPU vs GPU does not give identical numbers.  This script
#     1. moves every CPU-trained run of outputs/run_standard/runs to runs_cpu_<date>/
#     2. re-trains exactly those jobs with GPU workers only (same seeds, bundle, settings)
#     3. rebuilds results.csv and re-runs analysis/01..09 + robust summary
#  Torch-free rows (persistence, climatology, histgb) are left untouched.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
R=outputs/run_standard
BK="$R/runs_cpu_$(date +%F)"

echo "== step 1: move CPU-trained runs to $BK   ($(date '+%F %T'))"
python - "$R/runs" "$BK" <<'EOF'
import json, glob, os, shutil, sys
runs, bk = sys.argv[1], sys.argv[2]
os.makedirs(bk, exist_ok=True); n = 0
for mj in sorted(glob.glob(f"{runs}/*/metrics.json")):
    if json.load(open(mj)).get("config", {}).get("device") == "cpu":
        d = os.path.dirname(mj); shutil.move(d, os.path.join(bk, os.path.basename(d))); n += 1
        print("  moved", os.path.basename(d))
print(f"{n} CPU runs moved")
EOF

echo "== step 2: re-train on GPU only   ($(date '+%F %T'))"
python -m src.run_parallel --single_bundle --seeds 10 --epochs 60 --patience 15 --station_emb 8 \
       --gpu_workers 6 --cpu_workers 0 \
       --outdir "$R/runs" --results "$R/results.csv" --results_ext "$R/results_ext.csv"

echo "== step 3: analysis   ($(date '+%F %T'))"
export MTHAB_OUT=outputs MTHAB_FINAL_DIR="$R/runs" MTHAB_FINAL_FMT='A_mthab_mse_s{}' \
       MTHAB_BUNDLE=outputs/bundle_main.npz MTHAB_ANALYSIS_OUT=outputs/analysis
mkdir -p outputs/analysis_mixed_device_backup
cp outputs/analysis/*.csv outputs/analysis_mixed_device_backup/ 2>/dev/null || true
for s in analysis/01_reanchor_eventadj.py analysis/02_persistence_fullyear_events.py \
         analysis/03_onset_station_dataflow.py analysis/04_figures_table6.py \
         analysis/05_threshold_rules.py analysis/06_interpretation.py analysis/07_multitask_vs_single.py \
         analysis/08_validation_selection.py analysis/09_significance.py; do
  echo "--- $s" | tee -a outputs/analysis/log.txt
  python "$s" 2>&1 | tee -a outputs/analysis/log.txt || echo "!! $s failed" | tee -a outputs/analysis/log.txt
done
python -m src.robust_summary --results "$R/results.csv" --out "$R/robust" || true
echo "== done   ($(date '+%F %T'))"
