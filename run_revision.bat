@echo off
REM MT-HAB-G3D  ECOINF revision pipeline (Windows / Anaconda). Run from an "Anaconda Prompt":
REM     conda activate <env-with-torch>  &&  run_revision.bat          (GPU)
REM     run_revision.bat --smoke                                        (quick code check)
cd /d %~dp0
set PYTHONHASHSEED=0
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set SEEDS=10
set EPOCHS=60
set PATIENCE=15
if "%1"=="--smoke" (set SEEDS=1& set EPOCHS=2& set PATIENCE=1)

echo == stage 0: bundle
python -m src.data.dataset --daily outputs\daily_station_day.csv --out outputs\bundle_main.npz --impute ffill --mask_channel --T 10 --knn 2 --log_target --qc physical
echo KNN=2 STATION_KNN=0 LOG_TARGET=1 WINDOW=10 QC=physical IMPUTE=ffill3 MASK=1 > outputs\bundle_main.cfg

echo == stage 1: torch-free references
python -m src.run_reference --bundle outputs\bundle_main.npz --outdir outputs\run_standard\runs --results outputs\run_standard\results.csv --models persistence,climatology,histgb --seeds %SEEDS%

if "%1"=="--smoke" (
  python -m src.run_parallel --single_bundle --seeds 1 --epochs 2 --patience 1 --station_emb 8 --only lstm,mthab_mse,mthab_cls_only,mthab_reg_only --auto --outdir outputs\smoke\runs --results outputs\smoke\results.csv --results_ext outputs\smoke\results_ext.csv
  python smoke_check.py outputs\smoke\runs
  echo smoke test finished - see outputs\smoke
  goto :eof
)

echo == stage 2: neural matrix
python -m src.run_parallel --single_bundle --seeds %SEEDS% --epochs %EPOCHS% --patience %PATIENCE% --station_emb 8 --auto --outdir outputs\run_standard\runs --results outputs\run_standard\results.csv --results_ext outputs\run_standard\results_ext.csv

echo == stage 3: ablation
python -m src.run_ablation --device cuda --seeds %SEEDS% --epochs %EPOCHS% --patience %PATIENCE% --reuse_main outputs\run_standard\runs --workers 2

echo == stage 4: post-hoc analysis
set MTHAB_OUT=outputs
set MTHAB_FINAL_DIR=outputs\run_standard\runs
set MTHAB_FINAL_FMT=A_mthab_mse_s{}
set MTHAB_BUNDLE=outputs\bundle_main.npz
set MTHAB_ANALYSIS_OUT=outputs\analysis
python analysis\01_reanchor_eventadj.py
python analysis\02_persistence_fullyear_events.py
python analysis\03_onset_station_dataflow.py
python analysis\04_figures_table6.py
python analysis\05_threshold_rules.py
python analysis\06_interpretation.py
python analysis\07_multitask_vs_single.py
echo == done. tables: outputs\analysis\*.csv  figures: outputs\analysis\*.png
