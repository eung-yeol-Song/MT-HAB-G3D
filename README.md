# MT-HAB-G3D — Multi-Task Spatiotemporal Graph Learning for Coastal Harmful Algal Bloom Forecasting

Code, run manifests and result tables for the manuscript
*Multi-Task Spatiotemporal Graph Learning for Coastal Harmful Algal Bloom Forecasting*
(Song & Oh, submitted to *Ecological Informatics*, ECOINF-D-26-03683).

The repository contains everything needed to rebuild every table and figure of the
revised manuscript from the daily station table: data preparation, the proposed
MT-HAB-G3D model, thirteen comparison models, two single-task controls, the component /
candidate analysis, and the post-hoc statistics (seed-paired Wilcoxon with
Benjamini–Hochberg correction, moving-block bootstrap confidence intervals,
Diebold–Mariano tests, event-level detection, permutation importance).

## Contents

```
src/                 data pipeline, models, training, parallel runner
  data/dataset.py      daily table -> windowed bundle (physical QC, ffill<=3 d + mask channel, log1p target)
  models/mthab_g3d.py  proposed model (9-node variable graph, factorised + G3D pathways, two heads)
  models/mtgnn.py, cnn_lstm.py, baselines.py, reference.py   comparison models
  train.py             one run (model x seed), writes metrics.json / preds_test.npz / preds_val.npz
  run_parallel.py      full comparison matrix (142 runs), run_ablation.py  component analysis
analysis/            post-hoc analysis scripts 01-12 (tables, figures, supplement); numpy/pandas/scipy only
results/             CSV / TeX outputs that back the manuscript (see below)
run_revision.sh      one-command pipeline (Linux);  run_revision.bat  Windows / Anaconda
run_regpu.sh         re-run the reference matrix on GPU only (device unification, see "Determinism")
run_stage2.sh        full matrix re-trained on the three augmented input bundles (Supplement S1-S3)
smoke_check.py       5-10 min end-to-end check (1 seed, 2 epochs, 4 models)
requirements.txt     Python dependencies (PyTorch installed separately for your CUDA version)
```

### `results/`
| path | content |
|---|---|
| `run_standard/results.csv` | all 142 runs of the final comparison matrix (model, seed, parameters, runtime, every metric per lead time) |
| `run_standard/run_devices.csv` | device of every run (130 PyTorch runs on `cuda:0`, 12 torch-free reference runs) |
| `run_standard/reference_rows_summary.csv` | persistence / climatology / gradient-boosted trees |
| `ablation/ablation_results.csv`, `ablation_summary.csv`, `run_manifest.json` | component and candidate configurations (Table 7, Fig. 5) |
| `analysis/table*.csv`, `manuscript_tables.tex`, `key_numbers.txt` | Table 2–7 bodies exactly as typeset |
| `analysis/sample_ci.csv`, `dm_tests.csv`, `table5_pairs.csv` | bootstrap CIs, Diebold–Mariano, seed-paired tests |
| `analysis/validation_selection.csv` | pre-registered validation-only selection rule for candidate configurations |
| `analysis/onset_detection.csv`, `event_level_detection.csv`, `per_station_results.csv` | event-level and per-station results (Section 4.8, Supplement S5–S6) |
| `analysis/threshold_rules_summary.csv` | decision-threshold sensitivity (Supplement S4) |
| `analysis/perm_importance.csv`, `residual_mask_top_edges.csv`, `temporal_gates.csv` | model interpretation |
| `stage2_<doy|rthist|doy-rthist>/` | full matrix re-trained with added input channels (Supplement S1–S3) |

## Data

The daily station table (`daily_station_day.csv`: 17 coastal stations, 2015–2020, eight
water-quality variables from the NIFS real-time monitoring network plus the red-tide
report history) is **not redistributed here**; the monitoring data belong to the National
Institute of Fisheries Science (NIFS), Korea. The manuscript's Data availability section
states how to obtain it. Once the file is placed at `outputs/daily_station_day.csv`, the
pipeline below rebuilds the bundle and all runs.

Per-seed predictions and model weights (`preds_test.npz`, `model_state.pt`, ~0.5 GB in
total) are archived with the manuscript's data deposit (DOI to be added) rather than in
this repository.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA build
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Versions used for the manuscript: Python 3.10.12, PyTorch 2.13.0 (CUDA 13.0),
scikit-learn 1.7.2, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, Matplotlib 3.10.9,
one NVIDIA RTX 4090.

## Reproducing the manuscript

```bash
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
bash run_revision.sh --smoke        # 5-10 min sanity check -> "PASSED"
bash run_revision.sh                # full pipeline, ~6-8 h on one RTX 4090
```

`run_revision.sh` performs, in order:

1. **Bundle** – `src.data.dataset`: physical QC (pH ∉ [5,10], DO ≤ 0, EC/salinity ≤ 0, Chl-a < 0 set to
   missing), forward fill ≤ 3 days with a missingness mask channel, log1p Chl-a target, T = 10 day
   windows, correlation-based 2-NN variable graph; train 2015–2018 / validation 2019 / test 2020.
2. **Reference rows** – persistence, day-of-year climatology, gradient-boosted trees (10 seeds).
3. **Comparison matrix** – 13 models × 10 seeds + two single-task controls (`src.run_parallel`,
   60 epochs, patience 15, validation-selected thresholds per lead time).
4. **Component / candidate analysis** – `src.run_ablation` (spatial extension, station embedding,
   log target, T = 5, λ ∈ {0.2, 2.0}, persistence-anchored head, class-weighted BCE, added input channels).
5. **Analysis 01–12** – re-anchored event-adjacent subset, significance tests, CIs, event-level
   detection, interpretation, manuscript tables (`analysis/10`), figures 3–5 (`analysis/11`),
   supplement (`analysis/12`). Outputs go to `outputs/analysis/`.

Candidate configurations were compared on the 2019 validation year only
(`analysis/08_validation_selection.py`, S = 0.5·AUPRC + 0.5·R²; adopt iff ΔS > 0 and p < 0.05
over seeds). No candidate passed, so the final model is the reference configuration; test-set
numbers are reported as they come.

## Determinism

Runs are bit-reproducible given the seed **and the device**: the same seed trained on CPU and on
GPU gives different numbers. All 130 PyTorch runs behind the manuscript were trained on
`cuda:0` (`results/run_standard/run_devices.csv`). `run_revision.sh` therefore uses
GPU workers only; `--auto` (mixed CPU/GPU workers) is retained in `src.run_parallel` for quick
exploration but should not be used for publishable runs. `run_regpu.sh` documents how a mixed
matrix was unified.

## Citation

See `CITATION.cff`. Please cite the article once published; until then cite this repository.

## License

MIT (see `LICENSE`). The NIFS monitoring data are subject to their own terms and are not covered
by this license.
