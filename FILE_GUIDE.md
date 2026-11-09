# MOF-H2 Pipeline: Complete File Guide

## 1. Purpose of this guide

This document explains what the files in the repository contain, what each
file tells the reader, and how the files work together. It is a companion to
the scientific reports:

- `PROJECT_REPORT.md` explains the project history, datasets, models, and
  results.
- `REPRODUCTION_REPORT.md` gives the detailed reproduction procedure.
- `RESULTS_DISCUSSION.md` gives the numerical results and scientific
  interpretation.
- `README.md` is the short operational entry point.

The repository is a reproducible research workspace rather than only a Python
package. It contains source code, notebooks, simulation queues, RASPA
configuration, trained checkpoints, intermediate tables, final screening
tables, and the external ToBaCCo topology generator.

At the current revision there are 9,536 tracked files:

| Area | Approximate count | Role |
|---|---:|---|
| `src/` | 48 | Python and shell workflow code |
| `data/` | 5,764 | Inputs, intermediate results, queues, and final tables |
| `tobacco/` | 3,673 | Bundled ToBaCCo topology-generation software and databases |
| `notebooks/` | 10 | Sequential research and reproduction notebooks |
| `model_repo/` | 22 | GNN checkpoints, training histories, splits, and model results |
| `forcefield/` | 9 | RASPA force-field definitions |
| Other | 10 | Documentation, H2 molecule definition, and reference archive |

Some directories contain many repeated files. For example, `tobacco/templates`,
`tobacco/template_database`, and `data/` contain structure or result files
whose meaning is determined by their directory and naming convention. Those
families are described below rather than listing thousands of equivalent files
one by one.

## 2. End-to-end data flow

The main scientific flow is:

```text
hMOF labels/checkpoints
        |
        v
77 K GNN pretraining and audit
        |
        +--> ToBaCCo topology generation --> topology prediction/GCMC validation
        |
        +--> CoRE selection --> Zeo++/graph/Widom features --> 160 K GCMC
        |                                                     |
        |                                                     v
        |                                      160 K transfer models
        |
        +--> PoroNet labels + CIFs --> graph features --> 77 K adaptation
                                                              |
                                                              v
                                     baseline/adapted DOE screening and validation
```

The target convention used throughout the final workflow is:

```text
cold endpoint  = 77 K / 100 bar absolute uptake
warm endpoint  = 160 K / 5 bar absolute uptake
DOE capacity   = cold endpoint - warm endpoint
```

Unless a table explicitly says otherwise, uptake is in g/L.

## 3. Root-level files

### `.gitattributes`

Defines Git attributes for repository files. It is repository metadata, not
part of the scientific calculation.

### `.gitignore`

Excludes Python caches, virtual environments, temporary operating-system
files, and the local `data/poronet_cache_large/` feature cache. The enlarged
PoroNet cache is intentionally not committed because it is approximately
1.5 GB and can be regenerated from the PoroNet CIFs.

### `README.md`

The operational index for the project. It explains:

- the target definition and temperature/pressure convention;
- installation and environment setup;
- the hMOF, CoRE, ToBaCCo, PoroNet, and reference-MOF workflows;
- configurable graph limits such as `MOF_MAX_ATOMS`;
- how to reproduce screening and large-graph inference;
- where the important reports and output CSVs are located.

It is the best starting point for running the project, but it is not intended
to contain every historical experiment.

### `PROJECT_REPORT.md`

The broad project report. It records the motivation, datasets, simulation
conventions, model architectures, transfer-learning experiments, screening
results, limitations, and recommended production usage.

### `REPRODUCTION_REPORT.md`

The runbook. It is organized around commands and prerequisites: environment
setup, RASPA validation, feature generation, model training, fine-tuning,
screening, and result inspection. Use this file when the goal is to reproduce
the results rather than understand the scientific narrative.

### `RESULTS_DISCUSSION.md`

The numerical and scientific discussion report. It compares model variants,
documents domain shift, explains the PoroNet adaptation, discusses ranking
quality and false positives, and records the large-graph evaluation. It is the
most useful file for interpreting whether a model improvement is meaningful.

### `KNOWN_MOF_PREDICTIONS.md`

A concise report for named reference MOFs such as IRMOF, UiO, ZIF, MOF-74,
MIL, MOF-210, PCN, and Cu-BTC. It explains which structures were successfully
featurized, which exceeded the default atom guard, and where their predictions
are stored.

### `setup_env.sh`

Environment bootstrap script. It installs or prepares the Python-side
dependencies needed by the notebooks and source scripts. System-level RASPA
and Zeo++ availability may still be required separately.

## 4. Source code: model and feature core

### `src/mofnet.py`

Defines `MOFNet`, the graph neural network used for MOF hydrogen uptake.
The model consumes atom features, neighbor edges and distances, plus global
pore descriptors. It applies graph-attention/message-passing layers, pools
node embeddings to a structure embedding, combines that embedding with
normalized global descriptors, and predicts the configured target vector.
Checkpoint normalization statistics are stored with or alongside the model.

### `src/predict_topo.py`

The central CIF-to-feature and prediction utility for ToBaCCo and reference
MOFs. It:

1. parses a CIF with pymatgen;
2. creates atom features and a neighbor graph using the 4.5 Å cutoff;
3. computes or reads Zeo++ descriptors;
4. applies the configurable `MOF_MAX_ATOMS` guard;
5. loads the four-seed hMOF ensemble;
6. returns predictions, uncertainty, descriptor diagnostics, and status.

It demonstrates that the 600-atom value is a resource guard, not a GNN
architectural limit.

### `src/train_160.py`

Trains the real-CoRE transfer models for 160 K / 5 bar uptake. It compares
structure-only models with models augmented by Widom-derived `log(kH)` and
`qst0` features, evaluates held-out CoRE structures, and writes the saved
160 K ensemble checkpoints.

### `src/predict_160_checkpoint.py`

Standalone inference for the saved structure-plus-Widom 160 K checkpoint. It
loads a feature cache and optional Widom data, reconstructs the ensemble, and
prints or writes a 160 K prediction without rerunning training.

### `src/finetune_77.py`

Historical and CoRE-based 77 K transfer-learning script. It fine-tunes the
hMOF GAT on real CoRE GCMC labels, evaluates held-out structures, fits
baselines, and produces predictions used in earlier screening experiments.

### `src/finetune_77_poronet.py`

Fine-tunes the 77 K model on PoroNet/Tobacco labels. It creates the structural-
difficulty-stratified 80/20 split, trains the adapted ensemble, evaluates the
held-out set, and saves `finetune_77_poronet_ensemble.pth`.

### `src/finetune_130.py`

Exploratory transfer learning from the 77 K model to 130 K uptake at 5 and
100 bar using ToBaCCo structures and GCMC labels. Its outputs document the
intermediate-temperature experiment; it is not the final DOE model.

### `src/finetune_130_real.py`

Variant of the 130 K experiment using real-library structures and mixed
training data. It evaluates real cross-validation and external predictions.

### `src/pi_head.py`

Experiments with a physics-informed head on a frozen 77 K embedding. The head
predicts Langmuir/thermodynamic quantities such as Henry behavior, saturation,
and binding-related parameters before converting them to uptake. The associated
CSV files show whether this physically constrained head improved transfer.

### `src/featurise_core.py`

Creates graph and pore-descriptor feature records for selected CoRE MOFs.
The resulting cache is consumed by training, validation, and screening code.

### `src/featurise_real.py`

Featurizes the experimental real-library MOFs listed in `data/real_picks.csv`.
It writes the graph, global descriptors, and status information needed for
160 K transfer and external validation.

### `src/featurise_t130.py`

Featurizes generated ToBaCCo structures used in the 130 K experiment and
topology checks.

### `src/featurise_poronet.py`

Converts exported PoroNet CIFs into graph and Zeo++ feature caches. It supports
multiple workers, a custom cache location, and the enlarged atom limit used
for the 957 previously rejected structures.

### `src/export_poronet_cifs.py`

Extracts CIF text from the PoroNet structure archive/Feather source and writes
individual CIF files for local featurization.

### `src/prepare_poronet_benchmark.py`

Normalizes the published PoroNet labels into the project schema. It produces
paired 160 K, 77 K, and DOE-capacity columns and writes the benchmark manifest.

### `src/screen_poronet.py`

Screens the 1,990-structure PoroNet benchmark using the ordinary memory
path. It produces predictions for structures accepted by the default
600-atom guard and compares them with the published labels.

### `src/screen_poronnet_large.py`

Memory-bounded large-graph inference. It identifies structures rejected by the
original cache, loads only small chunks, evaluates baseline 77 K, adapted 77 K,
and structure-only 160 K ensembles, and writes:

`data/poronet_excluded_predicted_vs_actual.csv`

This script is why all 957 previously excluded structures could be evaluated
without loading every large graph into memory at once.

### `src/screen_predict.py`

The main CoRE screening predictor. It combines 77 K GNN predictions with a
160 K discharge estimate, calculates DOE capacity, propagates ensemble
uncertainty, and records applicability-domain and element flags.

Its output now also contains gravimetric forms of the uptake and DOE columns.
These are deterministic density-normalized values, not an independently
trained neural-network head.

### `src/gravimetric.py`

Shared conversion utility for adding gravimetric hydrogen uptake to existing
CSV tables. It converts volumetric g/L columns to both mg H2/g framework and
wt% using the row-specific framework density. It can read density directly
from a table or recover it from a feature-cache pickle by structure ID.

The conversion is exact for this repository's convention because
`src/extract.py` computes:

```text
g/L = (mg/g framework) * (framework density kg/m^3) / 1000
```

The utility therefore avoids training a redundant model for a quantity that is
already determined by the predicted volumetric uptake and the structure's
density.

### `src/screen_featurise.py`

Fast Zeo++ and graph featurization for the accessible, family-unique CoRE 2019
screening population.

### `src/predict_topo.py` and `src/zeo_fast.py`

These implement the standard and accelerated descriptor paths. `zeo_fast.py`
reduces Zeo++ sampling and avoids unnecessary CIF rewriting; its accuracy is
checked in `data/zeo_fast_validation.csv`.

## 5. Source code: simulation, selection, and analysis

### `src/core_meta.py`

Fast text-level parser for CoRE CIF metadata: atom count, volume, density,
metals, elements, cell dimensions, and hydrogen content.

### `src/core_res.py`

Reads and consolidates CoRE simulation results and descriptors. It also honors
the simulation-cell atom budget (`MOF_MAX_SIM_ATOMS`) and records why a
structure was accepted or rejected.

### `src/select_core.py`

Stage A CoRE selection. It chooses an accessible and de-duplicated set
stratified by pore size and density for expensive Widom calculations.

### `src/select_core_B.py`

Stage B selection. It reduces the Stage A set to a training subset while
oversampling strong binders using Henry and `qst0` information.

### `src/widom.py`

Runs or prepares Widom insertion calculations and extracts Henry/binding
features used by the 160 K transfer models.

### `src/prep_real.py`

Selects experimental MOFs from the RASPA library, prepares feature records, and
writes a 130 K simulation queue.

### `src/exp_isotherms.py`

Converts simulated 77 K isotherms from absolute uptake into excess uptake and
weight percent for comparison with experimental measurements.

### `src/av_feature.py`

Calculates H2-sized accessible volume using a 1.45 Å probe. This was used to
test whether inaccessible pockets explain prediction errors.

### `src/pocket_test.py`

Tests the inaccessible-pocket hypothesis by comparing geometric and H2-probe
accessible fractions with model residuals.

### `src/box_density.py`

Calculates the empty-box hydrogen density from GCMC runs. It supports the
H2-H2 convention audit.

### `src/audit_raw.py`

Re-reads values directly from raw RASPA output files and compares them with
stored result tables. This protects the reported metrics from table-generation
or transcription errors.

### `src/extract.py`

Low-level extraction helpers for simulation outputs, including data parsing,
PDB frames, and framework-related records.

### `src/blockfile.py`

Utility routines for block-file or simulation-output handling used by the
earlier workflow.

### `src/stage_io.py`

Moves intermediate results between notebook stages and packaged/reference
outputs. It supports restoring or unpacking stage artifacts.

### `src/build_queues.py`

Builds unified RASPA queue files from the packaged selection lists. It maps
stage names to simulation inputs and produces queue manifests.

### `src/compare.py`

Compares hMOF labels, GCMC outputs, and ML predictions for matched structures.

### `src/compare_160.py`

Tests whether the 77 K model can serve as a proxy for the 77 K-to-160 K DOE
swing. The results motivate explicit 160 K transfer learning.

### `src/compare_m2.py`

Compares alternative hydrogen-hydrogen interaction conventions at 100 bar.

### `src/compare_reference.py`

Compares results reproduced in the current environment with packaged results
from the original project or reference run.

### `src/compare_topo.py` and `src/compare_topo_m2.py`

Compare topology predictions against topology GCMC results, including the
alternative H2-H2 convention.

### `src/rq_any.sh`, `src/rq_gen.sh`, `src/run_box.sh`,
`src/run_box_t.sh`, `src/run_fh.sh`, `src/run_he.sh`, `src/run_widom.sh`

Shell helpers for submitting or launching RASPA jobs. The names identify the
simulation family: generic queues, generated structures, empty boxes,
temperature variants, force-field variants, helium calculations, and Widom
calculations.

## 6. Notebooks

The notebooks are numbered in execution order:

| Notebook | What it does |
|---|---|
| `00_setup_and_validation.ipynb` | Builds/checks the simulation environment and validates representative 77 K GCMC. |
| `01_hmof_models_and_label_audit.ipynb` | Audits hMOF labels and documents the XGBoost/GNN baseline models. |
| `02_gcmc_convention_and_core2024.ipynb` | Rebuilds GCMC, tests H2-H2 conventions, and checks CoRE data. |
| `03_topologies.ipynb` | Tests generalization to ToBaCCo topologies absent from hMOF. |
| `04_beyond77_130K.ipynb` | Tests the DOE swing and exploratory 130 K fine-tuning. |
| `05_core2019_160K_dataset.ipynb` | Builds the CoRE 2019 160 K dataset and transfer models. |
| `06_finetune77_and_screening.ipynb` | Fine-tunes 77 K and screens 7,206 CoRE structures. |
| `07_experiment_validation.ipynb` | Compares simulations and predictions with measured isotherms. |
| `08_audit_and_summary.ipynb` | Audits raw files and assembles the project summary. |
| `Hmof h2 modelling.ipynb` | Original hMOF modeling notebook: target definitions, GNN/XGBoost training, and label analysis. |

Notebook outputs are preserved as research evidence, while reusable logic is
gradually duplicated into `src/` so it can be run non-interactively.

## 7. Models and training artifacts

### `model_repo/models/*.pth`

- `gnn_full_random_s42.pth`, `s43`, `s44`, and `s45`: baseline hMOF GNN
  ensemble members.
- `gnn_full_building_block_s42.pth`: building-block split model.
- `gnn_full_topology_s42.pth`: topology split model.
- `gnn_graph_only_random_s42.pth`: graph-only ablation.
- `finetune_160_finetuneW_ensemble.pth`: fine-tuned 160 K model using Widom
  binding features.
- `finetune_160_structure_ensemble.pth`: structure-only 160 K model for cases
  without Widom features.
- `finetune_77_poronet_ensemble.pth`: PoroNet-adapted 77 K ensemble.

The checkpoints contain learned weights and normalization/target metadata.
They are loaded by the training and screening scripts rather than manually
edited.

### `model_repo/results/*.csv`

These files contain training histories, train/test splits, error-by-topology
tables, flagged labels, RASPA targets, and aggregate model comparisons. They
are diagnostics: they explain how a checkpoint was trained and where it fails,
but are not themselves model weights.

## 8. Force fields and molecular definitions

### `forcefield/UFF_FH/`

UFF with the project’s FH hydrogen interaction convention:
`force_field.def`, `force_field_mixing_rules.def`, and `pseudo_atoms.def`.

### `forcefield/UFF_FHm2/`

The alternative mass/interaction convention used for the validated RASPA
sample and the main `C19_BADJIC_clean` check.

### `forcefield/UFF_noFHH2/`

UFF variant without the explicit FH H2-H2 contribution. It is used in the
convention sensitivity analysis.

Each force-field directory must be used as a consistent set; mixing files
between directories changes the simulated physics.

### `molecules/local/H2single.def`

Local RASPA definition for the single-site hydrogen molecule used by the
project’s GCMC and Widom calculations.

### `reference/reference_runs.tgz`

Packaged reference outputs from earlier runs. It allows comparison against the
original results without rerunning every simulation.

## 9. Data directory

The `data/` directory contains both inputs and outputs. A CSV is not
necessarily a new independent dataset: many files are successive stages of
the same screening population.

### 9.1 Core metadata, selection, and features

- `core_meta.csv`: metadata for approximately 12,023 CoRE structures.
- `core_res.csv`: parsed pore descriptors such as LCD and PLD.
- `core_stageA.csv`: the approximately 1,000 Stage A selected structures.
- `core_stageB.csv`: the approximately 500 Stage B structures with Widom
  features and training split labels.
- `real_picks.csv`: real-library structures and featurization status.
- `t130_picks.csv`: generated structures used by the 130 K workflow.
- `screen_list.csv`: the approximately 7,206 CoRE structures prepared for
  screening.

### 9.2 Baseline validation and sensitivity tables

- `gcmc_validation.csv`: baseline GCMC agreement with reference labels.
- `h2h2_convention_test.csv`: comparison of H2-H2 interaction conventions.
- `core_residuals.csv`: per-structure residuals for selected CoRE checks.
- `core_sensitivity.csv`: sensitivity to density and void-fraction changes.
- `zeo_fast_validation.csv`: accurate versus fast Zeo++ descriptor comparison.
- `doe_swing_test.csv`: direct test of the 77 K-to-160 K DOE swing.
- `exp_gcmc_isotherms.csv`: simulated isotherm conversions.
- `exp_ml_predictions.csv`: model predictions for experimental reference MOFs.
- `pocket_test.csv`: accessible-volume and inaccessible-pocket analysis.

### 9.3 Training and transfer results

- `train_160_results.csv`, `train_160_doe.csv`, and
  `train_160_test_predictions.csv`: 160 K model comparisons and predictions.
- `finetune_130_results.csv`, `finetune_130_cv_predictions.csv`, and
  `finetune_130_external.csv`: generated-structure 130 K transfer results.
- `finetune_130_real_results.csv`, `finetune_130_real_cv.csv`, and
  `finetune_130_real_external.csv`: real-library 130 K transfer results.
- `finetune_77_results.csv` and `ft77_train.csv`: CoRE 77 K fine-tuning
  metrics and training membership.
- `pi_head_results.csv`, `pi_head_bulk_results.csv`, and their `*_doe.csv`
  companions: physics-informed-head experiments.

### 9.4 CoRE screening outputs

- `screen_core2019.csv`: baseline full CoRE screening predictions.
- `screen_core2019_ft77.csv`: screening using the adapted 77 K model.
- `screen_core2019_refined.csv`: post-filter/refined predictions.
- `screen_top10_gcmc.csv`, `screen_top20_ft_gcmc.csv`, and
  `screen_bottom5_gcmc.csv`: selected candidates and GCMC verification.
- `topo_predictions.csv` and `topo_predictions_clean.csv`: topology-library
  screening before and after cleaning.
- `topo_gcmc_picks.csv`, `topo_gcmc_validation.csv`, and
  `topo_gcmc_validation_m2.csv`: topology candidates and their simulation
  validation.

Most screening rows contain the same logical fields: structure ID, density,
void fraction, LCD/PLD, surface area, 77 K predictions, 160 K prediction,
DOE capacity, ensemble standard deviation, maximum descriptor z-score, and
applicability-domain flags.

### 9.5 PoroNet files

- `poronet_h2_manifest.csv`: provenance, paper/repository, units, and source.
- `poronet_h2_benchmark.csv`: 1,990 paired ground-truth endpoint labels.
- `poronet_screen_predictions.csv`: predictions for the 1,033 structures
  accepted under the default graph limit.
- `poronet_screen_predictions_77_finetuned.csv`: accepted-subset predictions
  after 77 K PoroNet fine-tuning.
- `poronet_validation_metrics.csv`: aggregate accepted-subset metrics.
- `poronet_77_finetune_test.csv` and
  `poronet_77_finetune_baseline_test.csv`: held-out adapted-versus-baseline
  metrics.
- `poronet_77_finetune_predictions.csv`: row-level held-out predictions.
- `poronet_excluded_predicted_vs_actual.csv`: row-level predictions for all
  957 structures originally excluded by the 600-atom guard. It includes actual
  labels, baseline and adapted predictions, uncertainty, errors, atom counts,
  pore descriptors, and applicability-domain status.
- `poronet_screen_predictions_gravimetric.csv`,
  `poronet_screen_predictions_77_finetuned_gravimetric.csv`, and
  `poronet_excluded_predicted_vs_actual_gravimetric.csv`: corresponding tables
  with `mg_per_g` and `wt_percent` columns for actual and predicted uptake,
  endpoint values, and DOE capacity.

The untracked `data/poronet_cache_large/` directory contains the regenerated
large graph features. It is ignored intentionally and is not required in Git
to understand the final CSV.

### 9.6 RASPA queue and log-control files

Files named `q_*.txt`, `queue*.txt`, and `done.txt` are simulation manifests.
Each row encodes a structure, pressure or stage, repetition counts, and a
target output directory. They are operational records, not model labels.

- `q_stage1.txt`, `q_stage2.txt`, `q_stage3.txt`: staged simulation queues.
- `q_w160.txt`, `q_160.txt`, `q_g160.txt`: 160 K/Widom queues.
- `q_ft77.txt`, `q_screen.txt`, `q_screenG.txt`: fine-tuning and screening
  queues.
- `q_exp.txt`, `q_real*.txt`, `q_top20.txt`, `q_bottom5.txt`: experimental,
  real-library, and candidate-validation queues.
- `queue_core.txt`, `queue_topo.txt`, `queue_fh.txt`, `queue_snap.txt`,
  `queue_rest.txt`, and `queue2.txt`: older or specialized queue stages.

### 9.7 JSON and text provenance files

- `gcmc_data.json` and `gcmc_data_v2.json`: structured descriptions of
  simulations, structures, descriptors, and validation status.
- `cases.json` and `cases_v2.json`: selected experiment/case definitions.
- `audit_dirs.txt`, `pores_out.txt`, and `done.txt`: audit and execution logs.
- `q_*.txt` and `queue*.txt`: queue manifests described above.
- `q_130*.txt`, `q_160.txt`, and `q_w160.txt`: temperature-specific queues.

### 9.8 User-provided or imported records

The `data/user_zip/` files preserve imported reference material:

- `core_top10.csv`: imported top-10 CoRE predictions;
- `hmof_targets.csv`: selected hMOF targets and predicted values;
- `raspa_reference_simin.txt`: reference RASPA simulation settings;
- `zeo_calibration.csv`: descriptor calibration statistics.

These files are useful for provenance and comparison but are not the primary
training database.

## 10. ToBaCCo directory

`tobacco/` is the bundled ToBaCCo 3.0 topology generator and its data. It is
an external computational component used to create MOF structures on topology
nets not necessarily represented in hMOF.

### Executable/source modules

- `tobacco.py`: main topology-generation orchestration.
- `configuration.py`: topology and generation configuration.
- `make_topologies.py`: topology construction.
- `place_bbs.py`: places building blocks.
- `write_cifs.py`: writes generated CIF structures.
- `ciftemplate2graph.py`: converts templates to graph representations.
- `SBU_geometry.py`, `Bio.py`: building-block and geometry utilities.
- `vertex_edge_assign.py`, `cycle_cocyle.py`, `adjust_edges.py`: graph/net
  operations.
- `remove_dummy_atoms.py`, `remove_net_charge.py`: CIF cleanup.
- `reindex.py`, `reindex_bb_cifs.py`: atom/index normalization.
- `scale.py`, `scale_animation.py`, `scaled_embedding2coords.py`: coordinate
  scaling and visualization helpers.
- `filter_templates.py`, `scrape_rcsr.py`, `topo_pore_analysis.py`:
  template filtering, RCSR access, and pore analysis.
- `run_one.sh`: example single-topology execution.

### Bundled databases and support files

- `RCSRnets-2019-06-01.cgd`: RCSR topology-net definitions.
- `nodes/`, `edges/`, `nodes_database/`, and `edges_database/`: topology and
  building-block databases.
- `template_2D_database/`, `template_database/`, and
  `template_database_old/`: template collections.
- `templates/`: generated or selected topology templates.
- `selected.txt`: selected topology list.
- `template_stats.json`: template statistics.
- `vertex_assignment.txt`: vertex assignment metadata.
- `tobacco_3.0_manual.pdf` and `.docx`: upstream user manuals.
- `README.md`, `LICENSE`, and `requirements.txt`: upstream documentation,
  licensing, and dependencies.

The thousands of files under these databases are inputs to the generator, not
independent ML results.

## 11. How to read the result tables

The most common columns have the following meanings:

| Column | Meaning |
|---|---|
| `uptake_160K_5bar_gL` | Simulated warm-endpoint absolute uptake |
| `uptake_77K_100bar_gL` | Simulated cold-endpoint absolute uptake |
| `deliverable_77K100_to_160K5_gL` | Actual DOE capacity |
| `pred_160K_5bar_gL` | Model warm-endpoint prediction |
| `pred_DOE_baseline_gL` | Baseline 77 K prediction minus predicted 160 K uptake |
| `pred_DOE_adapted_gL` | Adapted 77 K prediction minus predicted 160 K uptake |
| `*_std_gL` | Ensemble standard deviation, used as model uncertainty |
| `density`, `void_fraction`, `lcd`, `pld` | Global geometric descriptors |
| `n_atoms` | Number of atoms in the graph |
| `max_abs_descriptor_z` | Largest normalized distance from training descriptor statistics |
| `in_descriptor_domain` | Whether the descriptor z-score is below the project threshold |
| `status` | Whether parsing/featurization/prediction succeeded |

A high prediction should not automatically be treated as a good candidate.
Check the DOE value, ensemble uncertainty, atom count, domain flag, and
eventually validate the candidate with RASPA.

## 12. Recommended navigation

For a quick answer:

1. Start with `README.md`.
2. Use `data/*predictions*.csv` for row-level outputs.
3. Use `RESULTS_DISCUSSION.md` for interpretation.
4. Use `PROJECT_REPORT.md` for the full scientific record.
5. Use `REPRODUCTION_REPORT.md` to rerun a stage.
6. Use this guide when a file’s provenance or role is unclear.
