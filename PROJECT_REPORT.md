# MOF Hydrogen-Storage Screening Project Report

**Project:** Machine-learning and GCMC screening of metal-organic frameworks for hydrogen storage  
**Repository:** `mofh2-pipeline`  
**Report date:** 2026-11-06  
**Primary outputs:** predicted volumetric hydrogen uptake and DOE-style deliverable capacity  
**Units:** volumetric uptake in g/L unless explicitly stated otherwise

---

## 1. Executive summary

This project developed and validated a two-temperature workflow for screening
metal-organic frameworks (MOFs) for hydrogen storage:

```text
DOE deliverable capacity =
    uptake(77 K, 100 bar) - uptake(160 K, 5 bar)
```

The workflow combines:

1. **hMOF pretraining** on approximately 137,953 hypothetical MOFs at 77 K;
2. **graph neural network (GNN) inference** using framework graphs and pore
   descriptors;
3. **transfer learning** from the hMOF model to real CoRE-2019 structures;
4. **160 K Widom insertion and RASPA GCMC** data for the DOE warm-state target;
5. **PoroNet literature benchmarking** with paired 77 K and 160 K labels;
6. **domain-adaptation fine-tuning** of the 77 K model for the PoroNet/Tobacco
   structure distribution;
7. **uncertainty, applicability-domain, and false-positive analysis** to assess
   whether high-ranked candidates are reliable.

The main conclusions are:

- The original hMOF model is useful for ranking within its training domain but
  has a large domain shift on PoroNet/Tobacco structures.
- The 160 K transfer model is substantially improved by combining a fine-tuned
  GNN head with Widom features (`logkH` and `qst0`).
- The best CoRE 160 K model, `finetune+W`, achieved approximately **0.146 g/L
  MAE** on 75 held-out CoRE structures and **0.052 g/L MAE** on 11 external
  CoRE/hMOF structures.
- A single CoRE RASPA validation for `C19_BADJIC_clean` reproduced the archived
  uptake of approximately 2.53 g/L at 160 K / 5 bar within stochastic
  simulation variation.
- On the PoroNet benchmark, the structure-only 160 K model achieved **0.252 g/L
  MAE** over 1,033 featurized structures.
- The original 77 K hMOF model achieved **4.38 g/L MAE** on the held-out
  PoroNet split. Fine-tuning on a structural-difficulty-stratified 80/20 split
  reduced this to **0.60 g/L MAE**, with a Spearman correlation of **0.986**.
- The corresponding DOE deliverable-capacity error decreased from **6.66 g/L
  MAE** for the baseline ranking to **0.586 g/L MAE** after fine-tuning.
- The remaining major limitation is structure availability for the uploaded
  19,329-row `f1_*` label table. That file contains labels but not the matching
  CIF structures, so it cannot yet be screened with the graph models.
- QMOF was investigated but was not used for exact DOE validation because the
  publicly documented QMOF-GCMC labels expose working-capacity targets rather
  than the required paired absolute endpoints at 77 K / 100 bar and 160 K /
  5 bar.

---

## 2. Scientific objective and target definition

### 2.1 Storage objective

The project targets hydrogen storage in porous crystalline MOFs. The quantity
used for ranking is volumetric hydrogen uptake, reported as grams of hydrogen
per liter of framework:

```text
uptake: g H2 / L framework
```

The operating-point pair is intended to represent a cold high-pressure loading
state and a warmer lower-pressure discharge state:

| State | Temperature | Pressure | Role |
|---|---:|---:|---|
| Cold loading | 77 K | 100 bar | High-pressure uptake |
| Warm discharge | 160 K | 5 bar | Residual uptake |
| DOE swing | difference | difference | Working/deliverable capacity proxy |

The project uses the following equation:

```text
DOE swing = y77_100 - y160_5
```

where both terms are absolute volumetric uptake values in g/L.

### 2.2 Important distinction between targets

Several datasets in the project use different definitions. They must not be
mixed without an explicit conversion:

- hMOF pretraining labels: 77 K / 2 bar, 77 K / 100 bar, and
  `77 K / 100 bar - 77 K / 2 bar`;
- CoRE transfer labels: absolute 160 K / 5 bar uptake;
- DOE validation labels: absolute 77 K / 100 bar and 160 K / 5 bar uptake;
- PoroNet labels: paired absolute 77 K / 100 bar and 160 K / 5 bar uptake;
- QMOF-GCMC public metadata: primarily working-capacity targets over pressure
  or temperature ranges, not the exact DOE pair used here.

In particular, a 77 K working capacity from 5 to 100 bar is not equivalent to
absolute uptake at 77 K / 100 bar.

---

## 3. Repository and project lineage

The work combines two related repositories:

| Repository | Function |
|---|---|
| `ML-Screening-of-Metal-Organic-Frameworks-for-Hydrogen-Storage-material` | hMOF pretraining, baseline GNN checkpoints, initial screening and error analysis |
| `mofh2-pipeline` | RASPA validation, CoRE transfer learning, 160 K modeling, PoroNet benchmarking, and screening utilities |

The `mofh2-pipeline` repository is the authoritative reproduction repository
for the final transfer-learning and GCMC workflow. The hMOF repository remains
the source of the original pretraining context and baseline model development.

The major implementation areas are:

```text
src/mofnet.py                 GNN architecture
src/predict_topo.py           graph featurization and baseline inference
src/train_160.py              160 K transfer training
src/predict_160_checkpoint.py standalone 160 K checkpoint inference
src/finetune_77_poronet.py    PoroNet 77 K domain adaptation
src/featurise_core.py         CoRE feature generation
src/featurise_real.py         real-library feature generation
src/screen_predict.py         CoRE screening
src/screen_poronet.py         PoroNet screening
src/extract.py                RASPA uptake parser
src/widom.py                  Widom feature parser
src/run_fh.sh                 RASPA GCMC runner
src/run_widom.sh              Widom insertion runner
```

---

## 4. Data used in the project

### 4.1 hMOF pretraining data

The pretrained representation originated from the hMOF hydrogen-storage
dataset, derived from the MOFDB/Bobbitt-Snurr hypothetical-MOF release. The
working project description records approximately **137,953 hypothetical MOFs**.

The pretraining data contains:

- crystal structures represented as atom-bond/neighbor graphs;
- simulated hydrogen uptake at 77 K / 2 bar;
- simulated hydrogen uptake at 77 K / 100 bar;
- deliverable capacity defined as the difference between those two pressures;
- structural descriptors such as density, void fraction, surface area, LCD, and
  PLD.

The hMOF labels are volumetric values in g/L. They provide the initial
structure-to-uptake representation but do not directly provide the 160 K /
5 bar DOE target.

Baseline checkpoints used by the transfer and screening workflows include:

```text
model_repo/models/gnn_full_random_s42.pth
model_repo/results/gnn_full_random_s43.pth
model_repo/results/gnn_full_random_s44.pth
model_repo/models/gnn_full_random_s45.pth
```

These are ensemble members from different random seeds.

### 4.2 CoRE-2019 data

The real-MOF transfer experiment uses CoRE MOF 2019 structures and the
repository's packaged selection tables:

```text
data/core_meta.csv       CoRE metadata
data/core_res.csv        fast pore-size and geometry results
data/core_stageA.csv    1,000 Stage-A candidates
data/core_stageB.csv      500 Stage-B candidates
data/real_picks.csv      real-library selection/status table
```

The final CoRE selection contains:

- 425 CoRE structures used for training;
- 75 held-out CoRE structures used for the main test;
- 58 additional real-library structures used during transfer training;
- 11 external CoRE/hMOF structures used only for external evaluation.

The CoRE selection tables also preserve:

- train/test assignments;
- Widom values;
- structural descriptors;
- queue and simulation metadata;
- the 77 K validation subset.

### 4.3 RASPA simulation labels

The transfer data was assembled from multiple simulation stages:

| Stage | Calculation | Output |
|---|---|---|
| Widom | 160 K insertion | Henry coefficient `kH`, `qst0` |
| GCMC | 160 K / 5 bar | supervised warm-state uptake |
| GCMC validation | 77 K / 100 bar | DOE cold-state validation |

Queue files:

```text
data/q_w160.txt
data/q_stage2.txt
data/q_g160_known.txt
data/q_stage3.txt
```

Raw reference outputs are archived in:

```text
reference/reference_runs.tgz
```

### 4.4 PoroNet literature benchmark

The PoroNet benchmark was prepared from the open PoroNet release:

- Zheng, Gopalan, and Shi, *Journal of Chemical Theory and Computation*,
  DOI `10.1021/acs.jctc.6c00100`;
- 1,990 paired Tobacco/MOFDB structures;
- labels at 160 K / 5 bar and 77 K / 100 bar;
- reported standard errors for both GCMC endpoints;
- deliverable capacity calculated by subtraction.

Project artifacts:

```text
data/poronet_h2_benchmark.csv
data/poronet_h2_manifest.csv
data/poronet_cif/
data/poronet_cache/
```

Of the 1,990 paired structures:

- 1,033 were accepted by the current graph featurizer and evaluated;
- 957 exceeded the current 600-atom limit and were retained as uncovered
  labels rather than silently truncated.

The benchmark is especially valuable because it exposes the domain shift
between the hypothetical hMOF training distribution and real/generated
Tobacco structures.

### 4.5 Uploaded 22k-label database

The uploaded `clean_22k_gcmc.csv` contains:

- 19,329 unique structure IDs;
- complete labels at 77 K / 100 bar;
- complete labels at 160 K / 5 bar;
- deliverable capacity calculated from those endpoints;
- IDs using the `f1_*` naming convention.

Observed label ranges were approximately:

| Quantity | Minimum | Maximum |
|---|---:|---:|
| 77 K / 100 bar uptake | 0.0268 | 55.6329 |
| 160 K / 5 bar uptake | 0.0001 | 14.3955 |
| DOE deliverable capacity | 0.0257 | 53.1680 |

The CSV is labels-only. No matching `f1_*.cif` files were found in the
repository, and the linked Google Drive archive could not be reached from the
runtime. Consequently, this dataset was audited but not screened.

### 4.6 QMOF investigation

The QMOF repository and QMOF-GCMC metadata were inspected. The public
QMOF-GCMC documentation exposes hydrogen targets including:

- 5–100 bar working capacity at 77 K;
- 1–100 bar working capacity at 77 K;
- 298 K to 198 K working capacity over a pressure range.

The exact paired absolute targets required by this project were not identified.
The structure archive and labeled JSON were also hosted on external services
that were unreachable from the runtime. QMOF was therefore not included in
the final validation set.

---

## 5. Molecular simulation configuration

### 5.1 Production force field

The standard convention is `UFF_FHm2`:

- UFF framework Lennard-Jones parameters;
- single-site hydrogen molecule;
- hydrogen `epsilon/k = 36.7 K`;
- hydrogen `sigma = 2.958 A`;
- Feynman-Hibbs correction;
- hydrogen mass `2.016 u`;
- no framework partial charges;
- 12.8 A van der Waals cutoff;
- replicated cells selected to make the simulation box suitable for the cutoff.

Relevant files:

```text
forcefield/UFF_FHm2/
molecules/local/H2single.def
src/run_fh.sh
src/extract.py
src/widom.py
```

Alternative conventions were retained for diagnostics:

```text
forcefield/UFF_FH/
forcefield/UFF_noFHH2/
```

These alternatives are useful for sensitivity checks but are not the standard
production labels used in the transfer model.

### 5.2 RASPA validation sample

The structure `C19_BADJIC_clean` was simulated at 160 K / 5 bar using UFF_FHm2.
The run used approximately:

- 1,000 initialization cycles;
- 1,000 production cycles;
- replicated simulation cells selected by the repository script.

Two independent stochastic runs produced approximately:

```text
2.432 g/L
2.596 g/L
```

The archived/reference value was approximately:

```text
2.526 g/L
```

The results agree within expected Monte Carlo variation. This confirms that the
RASPA executable, force field, molecule definition, cell replication, parser,
and g/L conversion are operational for at least one real CoRE structure.

### 5.3 Simulation caveats

RASPA GCMC is stochastic. Repeated runs are not expected to be bit-identical.
Agreement should be judged using block uncertainty and the magnitude of the
observed variation, not exact equality.

The simulation convention must remain fixed when comparing labels. Changing the
hydrogen mass, Feynman-Hibbs treatment, cutoff, force field, or molecule
definition can shift the apparent uptake and invalidate direct comparisons.

---

## 6. Models and feature representations

### 6.1 GNN architecture

The core model is `MOFNet`, implemented in `src/mofnet.py`. It combines:

- graph-level information from the framework structure;
- atom-level and local-neighborhood features;
- global geometric descriptors;
- optional Widom-derived descriptors.

The model predicts normalized targets during training and converts them back to
g/L during inference. Multiple random-seed models are combined as an ensemble.
The ensemble mean is used as the central prediction and the ensemble standard
deviation is used as a model-spread uncertainty estimate.

### 6.2 Structural descriptors

The feature pipeline uses descriptors including:

- density;
- void fraction;
- accessible and non-accessible volume fractions;
- surface area;
- surface area per mass and per volume;
- largest cavity diameter (LCD);
- pore-limiting diameter (PLD);
- graph and elemental features;
- atom count and structure-level metadata.

Zeo++ is used for pore geometry and surface-area calculations. It is the major
runtime bottleneck in feature generation.

### 6.3 Widom features

The best 160 K model adds two normalized features:

```text
logkH = log10(Henry coefficient)
qst0  = zero-coverage isosteric heat of adsorption
```

The model checkpoint stores the normalization statistics needed to apply these
features consistently at inference time.

### 6.4 160 K model variants

The transfer experiment evaluated:

1. `henry` — Henry-law estimate;
2. `GBM` — gradient boosting model using structural features;
3. `GBM+W` — gradient boosting plus Widom features;
4. `probe` — frozen-GNN representation with a trainable probe/head;
5. `probe+W` — probe plus Widom features;
6. `finetune` — fine-tuned GNN/head without Widom features;
7. `finetune+W` — fine-tuned GNN/head plus Widom features.

The final reusable checkpoint is:

```text
model_repo/models/finetune_160_finetuneW_ensemble.pth
```

The structure-only alternative is:

```text
model_repo/models/finetune_160_structure_ensemble.pth
```

The `finetune+W` checkpoint contains four ensemble members and stores:

- base GNN state dictionaries;
- fine-tuned global-feature and prediction-head state dictionaries;
- target mean and standard deviation;
- Widom mean and standard deviation;
- feature names;
- training structure names;
- model metadata.

It is loaded by:

```text
src/predict_160_checkpoint.py
```

### 6.5 77 K domain-adapted model

The PoroNet adaptation starts from the hMOF 77 K ensemble and fine-tunes it
using paired PoroNet labels. The split was 80/20 and was stratified by
structural difficulty rather than target value alone.

A structure was flagged as difficult when it occupied a region such as:

- density below approximately one standard deviation from the hMOF domain;
- void fraction above approximately one standard deviation;
- LCD, PLD, or surface area more than approximately two standard deviations
  above the hMOF reference distribution.

The adapted checkpoint is:

```text
model_repo/models/finetune_77_poronet_ensemble.pth
```

---

## 7. 160 K transfer-learning results

### 7.1 Main CoRE test set

The main test set contains 75 held-out CoRE-2019 MOFs. Results are:

| Method | MAE (g/L) | Bias (g/L) | R2 | Spearman |
|---|---:|---:|---:|---:|
| Henry | 1.058 | 1.058 | -0.551 | 0.972 |
| GBM | 0.867 | 0.191 | 0.005 | 0.493 |
| GBM+W | 0.194 | 0.020 | 0.953 | 0.973 |
| Probe | 0.734 | 0.010 | 0.303 | 0.601 |
| Probe+W | 0.333 | 0.073 | 0.831 | 0.911 |
| Finetune | 0.630 | -0.020 | 0.428 | 0.717 |
| **Finetune+W** | **0.146** | **0.009** | **0.968** | **0.983** |

The main result is that Widom features are highly informative, but the best
performance comes from combining them with a fine-tuned structural
representation. Fine-tuning alone is not sufficient, and a purely tabular GBM
does not match the combined GNN-plus-Widom model.

### 7.2 External CoRE/hMOF set

The external set contains 11 structures not used for training:

| Method | MAE (g/L) | Bias (g/L) | R2 | Spearman |
|---|---:|---:|---:|---:|
| Henry | 0.119 | 0.119 | 0.883 | 0.991 |
| GBM | 0.336 | 0.256 | -0.688 | 0.273 |
| GBM+W | 0.144 | 0.101 | 0.748 | 0.682 |
| Probe | 0.444 | 0.390 | -1.083 | 0.318 |
| Probe+W | 0.388 | 0.384 | -0.600 | 0.155 |
| Finetune | 0.144 | 0.051 | 0.792 | 0.936 |
| **Finetune+W** | **0.052** | **0.033** | **0.973** | **0.936** |

The external result supports the choice of `finetune+W`, although the sample
size is small and should not be treated as a definitive uncertainty estimate.

### 7.3 DOE validation

The DOE validation set contains 41 structures for which the 77 K / 100 bar
endpoint and the 160 K / 5 bar endpoint were available.

The most informative comparison isolates the 160 K prediction error by using
the measured/validated 77 K endpoint and subtracting the best 160 K model:

| DOE estimate | n | MAE (g/L) | Bias (g/L) | Spearman |
|---|---:|---:|---:|---:|
| 77 K prediction minus fixed 1.54 g/L | 41 | 3.347 | 2.458 | 0.940 |
| 77 K prediction minus Henry | 41 | 3.430 | 0.290 | 0.915 |
| 77 K prediction minus GBM | 41 | 2.818 | 0.907 | 0.941 |
| 77 K prediction minus GBM+W | 41 | 3.166 | 1.156 | 0.927 |
| 77 K prediction minus Probe | 41 | 3.295 | 1.063 | 0.927 |
| 77 K prediction minus Probe+W | 41 | 3.270 | 1.042 | 0.926 |
| 77 K prediction minus Finetune | 41 | 3.202 | 1.310 | 0.928 |
| 77 K prediction minus Finetune+W | 41 | 3.199 | 1.201 | 0.923 |
| **77 K GCMC minus best 160 K model** | **41** | **0.134** | **0.007** | **0.998** |

This table separates the quality of the 160 K model from the separate problem
of predicting the 77 K cold endpoint. The 160 K transfer model itself is very
accurate on this validation subset; the larger total DOE error is dominated by
the 77 K endpoint prediction when it is also predicted rather than supplied.

### 7.4 Per-structure example

For `C19_BADJIC_clean`:

```text
Reference 160 K / 5 bar uptake: approximately 2.526 g/L
Fine-tune+W prediction:          approximately 2.290 g/L
```

The model prediction is within the expected scale of the stochastic GCMC
reference, while independent RASPA reruns gave approximately 2.432 and
2.596 g/L.

---

## 8. PoroNet benchmark and 77 K adaptation

### 8.1 Initial domain-shift result

The original hMOF model was applied to 1,033 PoroNet structures accepted by the
featurizer. The structure-only 160 K model remained comparatively accurate,
while the 77 K model showed a pronounced positive bias:

| Target | n | MAE (g/L) | RMSE (g/L) | Bias (g/L) | Spearman |
|---|---:|---:|---:|---:|---:|
| 160 K / 5 bar | 1,033 | 0.252 | 0.472 | -0.118 | 0.932 |
| 77 K / 100 bar | 1,033 | 4.599 | 5.435 | 3.668 | 0.809 |
| DOE deliverable | 1,033 | 4.613 | 5.509 | 3.785 | 0.805 |

The large 77 K error is not primarily a failure of the 160 K transfer model.
It is a domain-shift problem in the pretrained hMOF cold-state model.

### 8.2 Where the original model failed

The largest errors were concentrated in structures with combinations of:

- low framework density;
- high void fraction;
- large LCD;
- large PLD;
- high surface area;
- larger graph size and atom count.

Approximately 50.8% of the screened structures were outside the hMOF descriptor
domain by at least three standard deviations in at least one monitored
descriptor. This explains why a model that performs well on hMOF-like data can
systematically overpredict PoroNet/Tobacco uptake.

### 8.3 Fine-tuning split

The 80/20 adaptation split used:

- 826 training structures;
- 207 held-out test structures;
- structural-difficulty stratification;
- no use of the held-out labels during model selection.

The goal was to ensure that difficult low-density/high-void/large-pore regions
were represented in both train and test partitions.

### 8.4 Held-out fine-tuning results

| Target/model | n | MAE (g/L) | RMSE (g/L) | Bias (g/L) | Spearman |
|---|---:|---:|---:|---:|---:|
| Baseline 77 K / 100 bar | 207 | 4.380 | 5.125 | 3.547 | 0.775 |
| **Fine-tuned 77 K / 100 bar** | **207** | **0.601** | **1.161** | **0.107** | **0.986** |
| Baseline DOE deliverable | 207 | 6.657 | 9.168 | -3.165 | 0.053 |
| **Fine-tuned DOE deliverable** | **207** | **0.586** | **1.120** | **0.102** | **0.983** |

The adaptation therefore improved both absolute 77 K prediction and the final
DOE ranking. The result is a held-out evaluation, not a training-set fit.

### 8.5 Full PoroNet screening outputs

The main output files are:

```text
data/poronet_screen_predictions.csv
data/poronet_screen_predictions_77_finetuned.csv
data/poronet_77_finetune_predictions.csv
data/poronet_77_finetune_test.csv
data/poronet_77_finetune_baseline_test.csv
```

The fine-tuned screening table includes:

- structure ID;
- measured 160 K uptake and standard error;
- measured 77 K uptake and standard error;
- measured DOE capacity;
- baseline 77 K prediction;
- fine-tuned 77 K prediction;
- 160 K prediction and uncertainty;
- baseline and fine-tuned DOE predictions;
- hard-region flag.

### 8.6 False positives and ranking behavior

The original ranking contained meaningful false positives. Some candidates in
the predicted top 100 had true ranks around 500–560, while some true top
candidates were ranked below approximately 200–400. This is a practical
screening failure even when global correlation is moderate: a model can
correlate with the target while misordering the highest-value candidates.

After 77 K fine-tuning, the candidate ordering improved substantially. The
adapted ranking is therefore preferred for PoroNet/Tobacco-like structures,
provided the structure is within the accepted featurization and domain limits.

---

## 9. Exploratory 77 K and temperature-transfer work

The repository also contains exploratory artifacts for broader temperature
transfer and 77 K real-MOF adaptation:

```text
data/finetune_130_cv_predictions.csv
data/finetune_130_external.csv
data/finetune_130_real_cv.csv
data/finetune_130_real_external.csv
data/finetune_130_real_results.csv
data/finetune_130_results.csv
data/finetune_77_results.csv
```

These experiments tested alternative heads, residual corrections, linear
recalibration, and real-MOF transfer at temperatures other than the final
160 K target. They were useful for diagnosing transfer behavior, but the
production conclusions in this report are based on:

- the CoRE 160 K `finetune+W` model;
- the PoroNet 77 K fine-tuned model;
- the paired DOE validation results.

Some early recalibration attempts produced degenerate large biases on subsets
and were not selected for production. The final workflow uses direct
fine-tuning and explicit held-out evaluation instead of relying on a global
linear correction.

---

## 10. Screening workflow

For a new structure collection, the intended workflow is:

1. Obtain CIF files with stable IDs.
2. Generate graph and Zeo++ descriptors.
3. Reject or flag structures above the 600-atom graph limit.
4. Run the hMOF/77 K ensemble or the domain-adapted 77 K checkpoint.
5. Run the structure-only 160 K checkpoint if Widom data is unavailable.
6. If Widom features exist, use the `finetune+W` 160 K checkpoint.
7. Compute:

   ```text
   predicted DOE = predicted 77 K / 100 bar - predicted 160 K / 5 bar
   ```

8. Record ensemble standard deviation and descriptor-domain flags.
9. Rank candidates by predicted DOE capacity.
10. Select a small number of high-value, high-confidence candidates for
    independent RASPA validation.

The existing scripts are:

```text
src/screen_featurise.py
src/screen_predict.py
src/screen_poronet.py
src/predict_160_checkpoint.py
```

The screening output should retain all of the following rather than only the
rank:

- central prediction;
- ensemble uncertainty;
- 160 K uncertainty;
- descriptor z-score or applicability-domain flag;
- atom count;
- structure ID;
- whether the prediction used Widom features;
- whether the structure exceeded any featurizer limit.

### 10.1 Ranking policy

The model should be used primarily as a high-throughput ranking tool, not as a
replacement for GCMC. Recommended selection criteria are:

- high predicted DOE capacity;
- low ensemble uncertainty;
- in-domain descriptors;
- reasonable atom count and successful pore analysis;
- no invalid or missing Zeo++ measurements;
- diversity across chemical composition and topology.

The highest numerical prediction should not automatically be selected if it is
far outside the training domain or has high ensemble disagreement.

---

## 11. Reproducibility and checkpoint usage

### 11.1 Environment

The working environment used for the project included:

- Python;
- PyTorch;
- PyTorch Geometric;
- NumPy;
- pandas;
- SciPy;
- scikit-learn;
- pymatgen;
- XGBoost;
- CoolProp;
- RASPA2;
- Zeo++.

The local environment was prepared under:

```text
/root/repo-inspection/venv
```

### 11.2 Main reproduction commands

From the prepared project directory:

```bash
python src/featurise_core.py
python src/featurise_real.py
python src/train_160.py
```

For PoroNet:

```bash
python src/prepare_poronet_benchmark.py --source /path/to/PoroNet/GCMC_output
python src/export_poronet_cifs.py \
  --source /path/to/Selected_MOFs_for_H2.feather
python src/featurise_poronet.py --workers 8
python src/screen_poronet.py
python src/finetune_77_poronet.py
```

For saved 160 K checkpoint inference:

```bash
python src/predict_160_checkpoint.py \
  --checkpoint model_repo/models/finetune_160_finetuneW_ensemble.pth \
  --features /path/to/feature.pkl \
  --logkH -5.0 \
  --qst0 8.0
```

The exact command-line options should be checked with:

```bash
python src/predict_160_checkpoint.py --help
```

### 11.3 Checkpoint loading

The saved checkpoints contain metadata and should be loaded with:

```python
torch.load(path, weights_only=False)
```

The normalization stored in the checkpoint must be applied to external
features. Recomputing normalization from the screening set would introduce
data leakage and change the model definition.

---

## 12. Limitations and risks

### 12.1 Domain shift

The largest demonstrated risk is applying the hMOF model to structures outside
the hMOF descriptor distribution. Large pores, low density, high void fraction,
high surface area, and large atom counts are especially problematic.

### 12.2 Graph-size limit

The current graph featurizer rejects structures above 600 atoms. This affects
coverage and can bias a screening set toward smaller structures. The rejected
structures should be reported explicitly rather than removed without a count.

### 12.3 RASPA stochasticity

Finite-cycle GCMC results have statistical error. A model difference smaller
than the simulation uncertainty should not be overinterpreted.

### 12.4 Force-field dependence

UFF_FHm2 is a consistent screening convention, not a guarantee of experimental
accuracy. Different force fields, charges, hydrogen models, or quantum
corrections can change absolute uptake.

### 12.5 Dataset overlap

MOF databases may share structures or related derivatives. Final scientific
validation should deduplicate across hMOF, CoRE, PoroNet, QMOF, and any future
database before claiming independent generalization.

### 12.6 Small external test sets

The 11-structure external CoRE/hMOF result is encouraging but statistically
small. It should be treated as an external sanity check, not a replacement for
larger independent validation.

### 12.7 Labels without structures

The 19,329-row uploaded `f1_*` table cannot be used for graph inference without
the corresponding CIF files or an ID-to-CIF mapping. Labels alone permit
statistical inspection but not structure-based prediction.

### 12.8 QMOF target mismatch

QMOF working capacity is not automatically equivalent to the project's DOE
absolute-endpoint difference. It may be useful for a separate benchmark, but
it must be labeled with its exact thermodynamic definition.

---

## 13. Current artifact inventory

### Documentation

```text
README.md
REPRODUCTION_REPORT.md
PROJECT_REPORT.md
```

### Saved model checkpoints

```text
model_repo/models/gnn_full_random_s42.pth
model_repo/results/gnn_full_random_s43.pth
model_repo/results/gnn_full_random_s44.pth
model_repo/models/gnn_full_random_s45.pth
model_repo/models/finetune_160_finetuneW_ensemble.pth
model_repo/models/finetune_160_structure_ensemble.pth
model_repo/models/finetune_77_poronet_ensemble.pth
```

### Main evaluation tables

```text
data/train_160_results.csv
data/train_160_doe.csv
data/train_160_test_predictions.csv
data/poronet_validation_metrics.csv
data/poronet_77_finetune_baseline_test.csv
data/poronet_77_finetune_test.csv
data/poronet_77_finetune_predictions.csv
data/poronet_screen_predictions.csv
data/poronet_screen_predictions_77_finetuned.csv
data/doe_swing_test.csv
data/gcmc_validation.csv
data/topo_gcmc_validation.csv
```

### Data and simulation artifacts

```text
data/core_meta.csv
data/core_res.csv
data/core_stageA.csv
data/core_stageB.csv
data/real_picks.csv
data/poronet_h2_benchmark.csv
data/poronet_h2_manifest.csv
data/poronet_cif/
data/poronet_cache/
reference/reference_runs.tgz
```

---

## 14. Recommended next experiments

1. **Recover the matching CIF archive for the `f1_*` labels.**  
   This is the highest-value next step because it would provide 19,329
   structure/label pairs for an independent validation and screening study.

2. **Deduplicate all validation sources.**  
   Compare structural fingerprints or canonicalized CIFs across hMOF, CoRE,
   PoroNet, QMOF, and the `f1_*` database.

3. **Expand the 77 K fine-tuning set.**  
   Add independent paired real-MOF labels while preserving a strict held-out
   test set. Fine-tuning should target the low-density/high-void/large-pore
   failure regions rather than simply adding random structures.

4. **Relax or redesign the 600-atom graph limit.**  
   Large structures are common in porous materials and currently form an
   uncovered region. Possible solutions include hierarchical pooling,
   structure reduction, or a model that separates linker-level and framework
   level representations.

5. **Add calibrated uncertainty.**  
   Ensemble spread is useful but should be calibrated against absolute error.
   Conformal or validation-set calibration would make candidate selection more
   defensible.

6. **Run targeted RASPA validation.**  
   Select candidates spanning high predicted DOE, low uncertainty, high
   uncertainty, and out-of-domain flags. This tests whether the ranking and the
   uncertainty estimate behave as expected.

7. **Use QMOF only with an explicit target translation.**  
   If QMOF structures and labels become reachable, first determine whether the
   target is absolute endpoint uptake or working capacity. Do not combine the
   QMOF values with DOE labels until that definition is confirmed.

---

## 15. Final assessment

The project now has a reproducible end-to-end foundation:

- a documented hMOF pretrained GNN ensemble;
- real-MOF CoRE transfer data and simulation queues;
- a validated UFF_FHm2 RASPA setup;
- a reusable four-member 160 K `finetune+W` checkpoint;
- standalone checkpoint inference;
- a paired PoroNet benchmark;
- a structural-difficulty-stratified 77 K fine-tuned ensemble;
- quantitative evidence explaining the original false positives and domain
  failures;
- screening outputs with uncertainty and held-out validation metrics.

The strongest validated model combination is currently:

```text
77 K / 100 bar: PoroNet-adapted 77 K ensemble when screening PoroNet/Tobacco-like MOFs
160 K / 5 bar: CoRE-trained finetune+W ensemble when Widom features are available
DOE capacity:   difference of the two endpoint predictions, with domain and uncertainty flags
```

The workflow is ready for larger-scale screening once the corresponding CIF
structures are available for the labels-only databases. The predictions should
continue to be treated as ranking estimates requiring targeted RASPA or
experimental confirmation for final claims.
