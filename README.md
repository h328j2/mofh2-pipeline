# MOF Hydrogen-Storage Screening and DOE-Swing Transfer Learning

This repository contains a reproducible workflow for predicting gravimetric/volumetric
hydrogen uptake in metal-organic frameworks (MOFs), validating predictions with RASPA
grand-canonical Monte Carlo (GCMC), and estimating the DOE-style hydrogen swing:

```text
DOE swing = uptake(77 K, 100 bar) - uptake(160 K, 5 bar)
```

The consolidated numerical results and interpretation are in
[`RESULTS_DISCUSSION.md`](RESULTS_DISCUSSION.md). The broader project report is
in [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

The project has two related modeling stages:

1. **hMOF pretraining:** graph neural networks (GNNs) are trained on the large
   hypothetical-MOF dataset at 77 K and 2/100 bar.
2. **Real-MOF transfer learning:** the pretrained GNN representation is transferred
   to CoRE-2019 and real-library MOFs, with the target changed to 160 K / 5 bar.

The final saved transfer checkpoint is:

```text
model_repo/models/finetune_160_finetuneW_ensemble.pth
```

It is a four-member ensemble that predicts absolute H2 uptake in g/L at 160 K and
5 bar using structural descriptors plus two 160 K Widom features:

```text
logkH = log10(Henry coefficient)
qst0  = zero-coverage isosteric heat of adsorption
```

## Repository status

The repository includes the original hMOF checkpoints, CoRE selection tables,
RASPA inputs, reference simulation outputs, model-evaluation tables, and the
retrained 160 K fine-tuned ensemble checkpoint. GCMC is stochastic, so independently
rerunning a simulation will not produce bit-identical uptake values; agreement within
the reported block uncertainty and a few percent is expected.

## Repository structure

```text
.
├── README.md
├── setup_env.sh
├── data/
│   ├── core_meta.csv                 CoRE-2019 structure metadata
│   ├── core_res.csv                  Fast pore-size results
│   ├── core_stageA.csv               1,000 Widom-selected candidates
│   ├── core_stageB.csv               500 final CoRE candidates
│   ├── real_picks.csv                Real-library MOF selection/status table
│   ├── q_w160.txt                    160 K Widom queue
│   ├── q_stage2.txt                  160 K / 5 bar CoRE queue
│   ├── q_g160_known.txt              160 K / 5 bar library queue
│   ├── q_stage3.txt                  77 K / 100 bar validation queue
│   ├── train_160_results.csv         Transfer-model metrics
│   ├── train_160_doe.csv             DOE-swing validation metrics
│   ├── train_160_test_predictions.csv Per-sample predictions
│   ├── screen_core2019.csv           Initial CoRE screening results
│   ├── screen_core2019_refined.csv   Screening after Widom refinement
│   └── raspa_cif/                    Packaged MOF structures
├── forcefield/
│   ├── UFF_FHm2/                     UFF + Feynman-Hibbs, H2 mass 2.016 u
│   ├── UFF_FH/                       Alternative H2 mass convention
│   └── UFF_noFHH2/                   Classical, no H2-H2 Feynman-Hibbs
├── molecules/local/
│   └── H2single.def                  Single-site H2 molecule definition
├── model_repo/
│   ├── models/                       hMOF and saved transfer checkpoints
│   └── results/                      hMOF training histories/results
├── notebooks/
│   ├── 04_beyond77_130K.ipynb       130 K transfer experiments
│   ├── 05_core2019_160K_dataset.ipynb CoRE 160 K dataset and models
│   └── 06_finetune77_and_screening.ipynb Real-MOF 77 K fine-tuning/screening
├── src/
│   ├── mofnet.py                     GNN architecture
│   ├── predict_topo.py               Featurization and hMOF prediction utilities
│   ├── predict_160_checkpoint.py     Standalone inference from saved 160 K .pth
│   ├── featurise_core.py             CoRE graph/descriptor cache generation
│   ├── featurise_real.py             Real-library cache generation
│   ├── train_160.py                  160 K transfer models and checkpoint saving
│   ├── screen_predict.py             CoRE screening and 160 K prediction
│   ├── finetune_77.py                77 K real-MOF fine-tuning
│   ├── extract.py                    RASPA output parser
│   ├── widom.py                      Henry/qst0 parser
│   ├── run_fh.sh                     GCMC runner
│   ├── run_widom.sh                  Widom runner
│   └── rq_any.sh                     Resumable queue runner
├── reference/
│   └── reference_runs.tgz            Original raw RASPA outputs
└── tobacco/
    └── ToBaCCo resources used for generated-topology experiments
```

## Preliminary datasets

### 1. hMOF pretraining dataset

The preliminary model was trained using the hMOF hydrogen-storage dataset from the
MOFDB/Bobbitt-Snurr release:

- approximately **137,953 hypothetical MOFs**;
- simulated hydrogen isotherms at **77 K**;
- uptake labels at **2 bar** and **100 bar**;
- deliverable capacity defined as `y100 - y2`;
- values expressed as volumetric uptake in g/L;
- graph features from the framework structure;
- global descriptors including density, void fraction, surface area, LCD, and PLD.

The hMOF screening repository contains the original notebooks and the baseline
checkpoints used here. The checkpoints are:

```text
model_repo/models/gnn_full_random_s42.pth
model_repo/results/gnn_full_random_s43.pth
model_repo/results/gnn_full_random_s44.pth
model_repo/models/gnn_full_random_s45.pth
```

The hMOF models are useful for ranking and as the pretrained representation, but
they are not trained directly on the DOE 160 K / 5 bar target.

### 2. CoRE-2019 dataset

The real-MOF transfer dataset is based on CoRE MOF 2019 structures selected through
the packaged selection tables:

- `core_meta.csv` contains structure metadata;
- `core_res.csv` contains fast pore-size calculations;
- `core_stageA.csv` contains 1,000 Widom-selected candidates;
- `core_stageB.csv` contains 500 final candidates;
- `core_stageB.csv` has 425 training and 75 held-out test MOFs;
- 58 additional real-library MOFs are used for training;
- 11 external CoRE/hMOF structures are used as an external check.

The 160 K target data comes from:

- **Widom insertion at 160 K**, producing `kH` and `qst0`;
- **GCMC at 160 K / 5 bar**, producing the uptake target in g/L;
- **GCMC at 77 K / 100 bar** for a held-out subset, used to validate the DOE
  difference rather than to train the 160 K target.

The raw simulation queues are:

```text
data/q_w160.txt
data/q_stage2.txt
data/q_g160_known.txt
data/q_stage3.txt
```

The packaged `reference/reference_runs.tgz` archive contains the original raw
RASPA outputs used to reproduce the saved CSV results without rerunning every
simulation.

### 3. Published PoroNet literature benchmark

For an independent literature benchmark, the repository now includes
`data/poronet_h2_benchmark.csv`, prepared from the open PoroNet release:

- Zheng, Gopalan, and Shi, *J. Chem. Theory Comput.* 2026,
  DOI `10.1021/acs.jctc.6c00100`;
- 1,990 paired Tobacco/MOFDB structures;
- GCMC labels at **160 K / 5 bar** and **77 K / 100 bar**;
- standard errors for both labels;
- deliverable capacity computed as `uptake_77K_100bar_gL -
  uptake_160K_5bar_gL`.

The source spreadsheets and structure-selection list are published at
[`Shi-Research-Group/PoroNet`](https://github.com/Shi-Research-Group/PoroNet).
The corresponding CIFs are served by MOFDB and are not duplicated in this
repository. Once those CIFs are available, match their `tobmof-*` names to the
`mof` column and run the normal featurisation and checkpoint inference workflow.
The source metadata is recorded in `data/poronet_h2_manifest.csv`.

To regenerate the normalized benchmark table from the downloaded PoroNet
repository:

```bash
python src/prepare_poronet_benchmark.py \
  --source /path/to/PoroNet/GCMC_output
```

The PoroNet repository also publishes a Feather cache containing the selected
CIF text. Export and screen it with:

```bash
python src/export_poronet_cifs.py \
  --source /path/to/PoroNet/Pore_Graph_Generation/Selected_MOFs_for_H2.feather
python src/featurise_poronet.py --workers 8
python src/screen_poronet.py
```

The resulting `data/poronet_screen_predictions.csv` contains predictions and
ground-truth labels for every structure accepted by the current 600-atom graph
limit. Structures above that limit are retained in the benchmark label table
but are reported as uncovered rather than silently truncated. The limit is a
configurable resource guard rather than a fixed GNN requirement: set
`MOF_MAX_ATOMS` before featurization, for example
`MOF_MAX_ATOMS=1200 python src/featurise_poronet.py`. Larger graphs require
more RAM, edge-construction time, Zeo++ time, and inference time, so they
should be processed incrementally. The CoRE pore-screening prefilter uses the
same `MOF_MAX_ATOMS` setting and separately supports
`MOF_MAX_SIM_ATOMS` for the replicated simulation-cell atom budget.

All 957 structures excluded by the default limit have also been evaluated with
the memory-bounded `src/screen_poronnet_large.py` utility after featurization
at `MOF_MAX_ATOMS=5000`. The predicted-versus-actual results are in
`data/poronet_excluded_predicted_vs_actual.csv`. Reproduce the inference with:

```bash
MOF_MAX_ATOMS=5000 python src/screen_poronnet_large.py \
  --cache data/poronet_cache_large \
  --batch-size 2 --chunk-size 8
```

### 3. Real-library and external structures

`data/real_picks.csv` records the real-library structures used during transfer.
Rows marked `ok` are featurized and included in the 160 K training set. Rows marked
`too_large`, `duplicate`, or `error:*` are excluded by the scripts. The external
set is defined explicitly in `src/train_160.py` and is not used as training data.

## Simulation convention

The standard simulation convention is `UFF_FHm2`:

- RASPA2 Monte Carlo;
- UFF framework Lennard-Jones interactions;
- single-site H2 with `epsilon/k = 36.7 K`;
- H2 `sigma = 2.958 A`;
- Feynman-Hibbs correction using H2 mass `2.016 u`;
- no framework charges;
- 12.8 A van der Waals cutoff;
- replicated unit cells selected so perpendicular dimensions are suitable for
  the cutoff.

The standard transfer target is absolute uptake at **160 K / 5 bar**. DOE capacity
is obtained by subtracting that estimate from the 77 K / 100 bar prediction.

## Environment setup

The supplied setup script builds RASPA2 and Zeo++ and copies the project into a
working directory. It also installs the Python dependencies used by the notebooks.
For a normal Kaggle-style run:

```bash
export WORK=/kaggle/working/mofh2
bash setup_env.sh
```

The script prepares:

```text
$WORK/raspa/bin/simulate
$WORK/raspa/share/raspa/forcefield
$WORK/raspa/share/raspa/molecules/local
$WORK/raspa/share/raspa/structures/cif
$WORK/ext/zeopp-lsmo/zeo++/network
```

For local execution, make sure RASPA2, Zeo++, PyTorch, PyTorch Geometric,
NumPy, pandas, SciPy, scikit-learn, pymatgen, and xgboost are installed.

## Reproducing the real-CoRE transfer model

Run these commands from the prepared working directory (`$WORK`):

```bash
python featurise_core.py
python featurise_real.py
python train_160.py
```

The first two commands create:

```text
$WORK/core_cache/*.pkl
$WORK/real_cache/*.pkl
```

The training script reads the cached graph/descriptors and the RASPA outputs under
`runs_160/` and `runs_w160/`. It evaluates:

| Method | Description |
|---|---|
| `henry` | Henry-law estimate from Widom insertion |
| `GBM` | Gradient boosting from structural descriptors |
| `GBM+W` | Gradient boosting from descriptors plus `logkH` and `qst0` |
| `probe` | Ridge regression on frozen GNN representations |
| `probe+W` | Frozen representation plus Widom features |
| `finetune` | Retrained global MLP and target head |
| `finetune+W` | Fine-tuned model plus Widom features |

The best validated method in the packaged run was `finetune+W`. The script writes:

```text
train_160_results.csv
train_160_doe.csv
train_160_test_predictions.csv
model_repo/models/finetune_160_finetuneW_ensemble.pth
```

The checkpoint contains four ensemble members, the frozen base GNN weights, the
fine-tuned global MLP and head weights, target normalization, Widom normalization,
feature names, and the training-MOF names.

## Using the saved model to screen MOFs

### Option A: screen the packaged CoRE-2019 library

The repository already contains the CoRE screening list and cached/result tables.
The intended pipeline is:

1. Featurize each candidate structure with graph features and Zeo++ descriptors.
2. Predict 77 K / 2 bar and 77 K / 100 bar with the hMOF ensemble.
3. Predict 160 K / 5 bar with the transferred model.
4. Compute:

   ```text
   predicted_doe_capacity = predicted_77K_100bar - predicted_160K_5bar
   ```

5. Sort candidates by predicted DOE capacity.
6. Inspect uncertainty, applicability-domain flags, element coverage, and pore
   descriptors before selecting candidates for GCMC confirmation.

For the original structure-only CoRE pass:

```bash
python screen_predict.py
```

For the Widom-refined pass, after Widom outputs exist under `runs_w160/`:

```bash
python screen_predict.py refine
```

The results are written to:

```text
screen_core2019.csv
screen_core2019_refined.csv
```

Important columns include:

```text
gnn77_y100       predicted 77 K / 100 bar uptake
p160             predicted 160 K / 5 bar uptake
doe_dc           predicted DOE swing
p160_source      structure-only or GAT + Widom
gnn77_y100_std   ensemble uncertainty
in_domain        descriptor/element applicability-domain flag
```

The original pipeline uses a structure-only 160 K estimate for all candidates and
replaces it with the more accurate `GAT + Widom` estimate for candidates with a
completed 160 K Widom calculation. For a new MOF, the recommended process is to
screen structurally first, run Widom only for shortlisted candidates, then rerun
the refined prediction before expensive 160 K / 5 bar GCMC.

For direct inference from a saved feature pickle and the persistent checkpoint:

```bash
python src/predict_160_checkpoint.py \
  /path/to/feature.pkl \
  --logkH -5.2 \
  --qst0 8.5 \
  --output prediction_160k.csv
```

For multiple candidates, use a CSV with `name`, `logkH`, and `qst0`:

```bash
python src/predict_160_checkpoint.py \
  /path/to/a.pkl /path/to/b.pkl \
  --widom-csv widom_features.csv \
  --output predictions_160k.csv
```

The script reports predicted 160 K / 5 bar uptake and four-member ensemble
standard deviation. It does not predict the 77 K / 100 bar term or DOE swing;
combine it with the baseline hMOF prediction to calculate DOE capacity.

### Option B: screen a new MOF collection

To screen new CIF files, place them in the working RASPA structure directory and
create a candidate table following the schema of `data/screen_list.csv`:

```text
file,name,family,n_atoms,density,metals,cells,lcd_raw,pld_raw
```

Then:

1. Run the same Zeo++ and graph featurization used by `screen_featurise.py`.
2. Store one feature cache pickle per candidate under `screen_cache/`.
3. Ensure each feature record contains:
   - `x`
   - `edges`
   - `dists`
   - `void_fraction`
   - `surface_area_m2g`
   - `surface_area_m2cm3`
   - `lcd`
   - `pld`
   - `density`
   - `elements`
4. Run `screen_predict.py` after adapting its candidate-table input if the new
   names are not in `screen_list.csv`.
5. For shortlisted structures, run 160 K Widom and then:

   ```bash
   python screen_predict.py refine
   ```

The saved `.pth` checkpoint is the persistent 160 K model artifact. A prediction
application must reconstruct the same graph tensors and global descriptors, apply
the checkpoint's `widom_mean`/`widom_std`, use the stored four ensemble members,
and average their predictions. The checkpoint target is **absolute 160 K / 5 bar
uptake**, not DOE capacity; DOE capacity still requires a compatible 77 K / 100 bar
prediction.

## Validating a candidate with RASPA

Use the supplied runner for a single validation simulation:

```bash
TEMP=160 FF=UFF_FHm2 bash run_fh.sh \
  C19_BADJIC_clean 5 1000 1000 "4 1 3" runs_160/C19_BADJIC_clean_5
```

The output is under:

```text
runs_160/C19_BADJIC_clean_5/Output/System_0/*.data
```

The parser in `src/extract.py` converts absolute loading from mg/g and framework
density to g/L. Reruns are expected to differ modestly because GCMC is stochastic.
Use more cycles and independent repeats for final validation.

For queued simulations:

```bash
export WORK=/kaggle/working/mofh2
export NPROC=4
bash rq_any.sh queues/05_core2019_160K.txt 11
```

## Model limitations

- The hMOF pretraining domain is hypothetical and does not cover every real-MOF
  chemistry.
- The 160 K transfer target is volumetric uptake in g/L; it is not gravimetric
  wt% unless explicitly converted using framework density/mass conventions.
- The Widom-refined model requires reliable 160 K Widom calculations.
- Candidates with unseen elements or extreme descriptor z-scores are flagged by
  `in_domain` and should not be treated as equally reliable.
- Screening predictions are rankings and estimates, not substitutes for final
  GCMC or experimental validation.
- RASPA labels and predictions are stochastic and depend on force field, cell
  replication, cycle counts, and molecule definition.

## Main output files

```text
data/train_160_results.csv
data/train_160_doe.csv
data/train_160_test_predictions.csv
data/finetune_130_results.csv
data/finetune_130_real_results.csv
data/finetune_77_results.csv
data/screen_core2019.csv
data/screen_core2019_refined.csv
model_repo/models/finetune_160_finetuneW_ensemble.pth
```

The CSV files are reproducibility summaries. The `.pth` file is the reusable
fine-tuned model artifact.
