#!/usr/bin/env bash
# =====================================================================================
#  Unify the three stage-2 matrices (Supplementary Tables S1-S3) on one device, the same
#  way run_regpu.sh did for the reference matrix.  `run_parallel --auto` had placed 35-38
#  of the 142 jobs of each matrix on CPU workers (incl. the proposed model's seeds 6-9).
#  For each input-channel bundle this script
#     1. moves every CPU-trained run of outputs/stage2_<f>/run_standard/runs to
#        runs_cpu_<date>/ (with the old manifest and the old analysis CSVs)
#     2. re-trains exactly those jobs with GPU workers only (same seeds, bundle, settings)
#     3. rebuilds results.csv and re-runs the stage-2 analysis + robust summary
#  The gradient-boosted-tree runs that a CPU worker had executed (seeds 6-9) carry the same
#  device label and are repeated as well; they do not use PyTorch and reproduce bit-identically.
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
STAMP=$(date +%F)

for f in ${FEATS:-doy rthist doy-rthist}; do
  B="outputs/bundle_ab_k2_s0_l1_T10_qc_f${f}.npz"
  D="outputs/stage2_${f}"; R="$D/run_standard"; BK="$R/runs_cpu_${STAMP}"
  [ -f "$B" ] || { echo "missing bundle $B"; exit 1; }

  echo "== [$f] step 1: move CPU-trained runs to $BK   ($(date '+%F %T'))"
  python - "$R/runs" "$BK" <<'EOF'
import json, glob, os, shutil, sys
runs, bk = sys.argv[1], sys.argv[2]
os.makedirs(bk, exist_ok=True); n = 0
if os.path.exists(f"{runs}/run_manifest.json"):
    shutil.copy2(f"{runs}/run_manifest.json", f"{bk}/run_manifest_mixed_device.json")
for mj in sorted(glob.glob(f"{runs}/*/metrics.json")):
    if json.load(open(mj)).get("config", {}).get("device") == "cpu":
        d = os.path.dirname(mj); shutil.move(d, os.path.join(bk, os.path.basename(d))); n += 1
        print("  moved", os.path.basename(d))
print(f"{n} CPU runs moved")
EOF
  mkdir -p "$D/analysis_mixed_device_backup"
  cp "$D"/analysis/*.csv "$D/analysis_mixed_device_backup/" 2>/dev/null || true
  cp "$R/results.csv" "$D/analysis_mixed_device_backup/results_mixed_device.csv"
  cp -r "$R/robust" "$D/analysis_mixed_device_backup/robust" 2>/dev/null || true

  echo "== [$f] step 2: re-train on GPU only   ($(date '+%F %T'))"
  python -m src.run_parallel --bundle "$B" --seeds 10 --epochs 60 --patience 15 --station_emb 8 \
         --gpu_workers 6 --cpu_workers 0 \
         --outdir "$R/runs" --results "$R/results.csv" --results_ext "$R/results_ext.csv"

  echo "== [$f] step 3: analysis   ($(date '+%F %T'))"
  export MTHAB_OUT="$D" MTHAB_FINAL_DIR="$R/runs" MTHAB_FINAL_FMT='A_mthab_mse_s{}' \
         MTHAB_BUNDLE="$B" MTHAB_ANALYSIS_OUT="$D/analysis"
  for s in analysis/01_reanchor_eventadj.py analysis/02_persistence_fullyear_events.py \
           analysis/03_onset_station_dataflow.py analysis/04_figures_table6.py \
           analysis/05_threshold_rules.py analysis/06_interpretation.py analysis/07_multitask_vs_single.py \
           analysis/09_significance.py; do
    echo "--- $s" | tee -a "$D/analysis/log.txt"
    python "$s" 2>&1 | tee -a "$D/analysis/log.txt" || echo "!! $s failed" | tee -a "$D/analysis/log.txt"
  done
  python -m src.robust_summary --results "$R/results.csv" --out "$R/robust" || true
  echo "== [$f] done   ($(date '+%F %T'))"
done
echo "== stage-2 GPU unification finished for: ${FEATS:-doy rthist doy-rthist}   ($(date '+%F %T'))"
