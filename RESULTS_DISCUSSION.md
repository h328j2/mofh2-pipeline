# Consolidated Results and Discussion

**Project:** MOF hydrogen-storage machine learning and RASPA screening  
**Repository:** `mofh2-pipeline`  
**Compilation date:** 2026-11-07

This document consolidates the numerical results generated during the project.
It is intended to be read together with `PROJECT_REPORT.md`, which describes
the data lineage, file layout, model implementation, and reproduction commands
in greater detail.

Unless stated otherwise, uptake is volumetric hydrogen uptake in g/L. The
project's DOE-style target is:

```text
DOE deliverable capacity =
    uptake(77 K, 100 bar) - uptake(160 K, 5 bar)
```

The results are separated into:

1. the production 160 K CoRE transfer model;
2. exploratory temperature-transfer experiments;
3. 77 K domain adaptation on PoroNet;
4. screening and ranking validation;
5. RASPA, force-field, and descriptor validation;
6. scientific interpretation, limitations, and recommendations.

---

## 1. Overall findings

The project produced three main scientific conclusions.

### 1.1 The warm-state model is highly recoverable by transfer learning

The hMOF GNN representation can be transferred from 77 K hydrogen labels to
real-MOF 160 K / 5 bar uptake. The best model combines:

- fine-tuning of the GNN/global prediction head;
- 160 K Widom insertion features;
- a four-member ensemble.

The resulting `finetune+W` model achieved:

| Evaluation set | n | MAE | Bias | R2 | Spearman |
|---|---:|---:|---:|---:|---:|
| CoRE-2019 held-out test | 75 | 0.146 | 0.009 | 0.968 | 0.983 |
| External CoRE/hMOF set | 11 | 0.052 | 0.033 | 0.973 | 0.936 |

This is the strongest validated single-endpoint model in the project.

### 1.2 The original cold-state model has a substantial domain shift

The pretrained 77 K hMOF model performs substantially worse on PoroNet/Tobacco
structures than on hMOF-like structures. On 1,033 featurized PoroNet structures:

| Target | MAE | RMSE | Bias | Spearman |
|---|---:|---:|---:|---:|
| 160 K / 5 bar | 0.252 | 0.472 | -0.118 | 0.932 |
| 77 K / 100 bar | 4.599 | 5.435 | +3.668 | 0.809 |
| DOE deliverable | 4.613 | 5.509 | +3.785 | 0.805 |

The dominant failure is systematic overprediction of the 77 K endpoint, not
failure of the 160 K model.

### 1.3 Targeted 77 K fine-tuning fixes most of the demonstrated failure

Fine-tuning on an 80/20 PoroNet split, stratified to include difficult
structural regions, reduced held-out errors to:

| Target | Baseline MAE | Fine-tuned MAE | Baseline Spearman | Fine-tuned Spearman |
|---|---:|---:|---:|---:|
| 77 K / 100 bar | 4.380 | 0.601 | 0.775 | 0.986 |
| DOE deliverable | 6.657 | 0.586 | 0.053 | 0.983 |

The improvement is large enough to change which structures should be selected
for downstream GCMC validation.

---

## 2. Dataset and experiment matrix

### 2.1 hMOF pretraining

The original representation was trained on approximately 137,953 hypothetical
MOFs with simulated hydrogen uptake at:

- 77 K / 2 bar;
- 77 K / 100 bar;
- 77 K deliverable capacity, calculated as the pressure difference.

The models use framework graphs and global descriptors such as density, void
fraction, surface area, LCD, and PLD. The hMOF ensemble is the starting point
for all subsequent transfer experiments.

### 2.2 CoRE 160 K transfer data

The real-MOF transfer study uses:

- 425 CoRE structures for training;
- 75 held-out CoRE structures for testing;
- 58 real-library structures;
- 11 external CoRE/hMOF structures;
- 160 K Widom `kH` and `qst0`;
- 160 K / 5 bar GCMC uptake;
- a 77 K / 100 bar subset for DOE validation.

### 2.3 PoroNet benchmark

The PoroNet benchmark contains 1,990 paired structures with labels at:

- 160 K / 5 bar;
- 77 K / 100 bar;
- the difference between the two endpoints.

The current graph featurizer accepted 1,033 structures and rejected 957 because
they exceeded the 600-atom limit. The rejected structures remain important:
they define an uncovered part of the screening population.

### 2.4 Uploaded labels-only database

The uploaded `clean_22k_gcmc.csv` contains 19,329 unique IDs and complete DOE
endpoint labels. The corresponding CIF structures were not available, so this
database was audited but not used for structure-based inference.

---

## 3. Production 160 K transfer results

### 3.1 Model comparison on 75 held-out CoRE structures

The complete 160 K comparison is:

| Method | MAE | Bias | R2 | Spearman |
|---|---:|---:|---:|---:|
| Henry estimate | 1.058 | +1.058 | -0.551 | 0.972 |
| GBM descriptors | 0.867 | +0.191 | 0.005 | 0.493 |
| GBM + Widom | 0.194 | +0.020 | 0.953 | 0.973 |
| Frozen-GNN probe | 0.734 | +0.010 | 0.303 | 0.601 |
| Frozen-GNN probe + Widom | 0.333 | +0.073 | 0.831 | 0.911 |
| Fine-tuned GNN | 0.630 | -0.020 | 0.428 | 0.717 |
| **Fine-tuned GNN + Widom** | **0.146** | **+0.009** | **0.968** | **0.983** |

### 3.2 Interpretation

Three patterns are clear.

First, `GBM+W` is much better than the descriptor-only GBM. The Widom
calculation supplies information about low-loading gas-framework interactions
that cannot be reconstructed reliably from a small set of geometric descriptors.

Second, fine-tuning the GNN without Widom features is not enough. The structure
representation transfers useful information, but the 160 K endpoint depends on
adsorption energetics that are only partially encoded in the graph and pore
descriptors.

Third, combining both sources is complementary. The GNN captures framework
structure and geometry, while `logkH` and `qst0` provide direct information
about gas adsorption thermodynamics. This is why `finetune+W` is the production
choice.

### 3.3 External evaluation on 11 structures

The external results are:

| Method | MAE | Bias | R2 | Spearman |
|---|---:|---:|---:|---:|
| Henry estimate | 0.119 | +0.119 | 0.883 | 0.991 |
| GBM descriptors | 0.336 | +0.256 | -0.688 | 0.273 |
| GBM + Widom | 0.144 | +0.101 | 0.748 | 0.682 |
| Frozen-GNN probe | 0.444 | +0.390 | -1.083 | 0.318 |
| Frozen-GNN probe + Widom | 0.388 | +0.384 | -0.600 | 0.155 |
| Fine-tuned GNN | 0.144 | +0.051 | 0.792 | 0.936 |
| **Fine-tuned GNN + Widom** | **0.052** | **+0.033** | **0.973** | **0.936** |

The external sample is small, but the same model that wins on the 75-structure
test set also wins externally. That consistency supports the model selection.

### 3.4 DOE endpoint isolation

The 41-structure DOE subset was used to separate warm-state error from
cold-state error. When the measured/validated 77 K endpoint is supplied and
only the 160 K term is predicted, the best 160 K model gives:

```text
MAE:      0.134 g/L
Bias:     0.007 g/L
Spearman: 0.998
```

When both endpoints are predicted, the total DOE error is much larger. This
shows that DOE performance is a compound metric:

```text
DOE error = cold-state error - warm-state error
```

Improving the 160 K model alone cannot solve the full DOE ranking problem if the
77 K endpoint remains systematically biased.

### 3.5 Example structure

For `C19_BADJIC_clean`:

```text
Archived 160 K / 5 bar reference: approximately 2.526 g/L
Fine-tune+W prediction:           approximately 2.290 g/L
Independent RASPA reruns:         approximately 2.432 and 2.596 g/L
```

The prediction is close to the stochastic simulation range. This structure was
also used to verify checkpoint reloading and standalone inference.

---

## 4. Exploratory 130 K transfer results

The 130 K experiments were exploratory rather than the final production target.
They tested how much of the hMOF representation transfers to another
temperature and pressure regime.

### 4.1 ToBaCCo cross-validation

For 157 ToBaCCo structures in five-fold cross-validation:

| Method | 130 K / 5 bar MAE | 130 K / 100 bar MAE |
|---|---:|---:|
| Descriptor-only GBM | 0.374 | 1.052 |
| Scaled 77 K predictions | 0.226 | 0.757 |
| Frozen-GNN linear probe | 0.215 | 0.631 |
| **Fine-tuned GNN** | **0.205** | **0.484** |

The fine-tuned GNN is best on both 130 K targets in the internal ToBaCCo
cross-validation. The improvement is larger at 100 bar, where a simple
temperature scaling rule is less reliable.

### 4.2 External 130 K evaluation

On the 11-structure external CoRE/hMOF set:

| Method | 130 K / 5 bar MAE | 130 K / 100 bar MAE |
|---|---:|---:|
| Descriptor-only GBM | 0.399 | 2.093 |
| Scaled 77 K predictions | 0.405 | 2.168 |
| Frozen-GNN linear probe | 0.421 | 1.984 |
| Fine-tuned GNN | 0.412 | 2.119 |

The internal cross-validation improvement does not transfer cleanly to the
small external set. This is an important warning: validation within one
generated-topology distribution can overstate performance on real structures.

### 4.3 Adding real-library structures

For the 58 real-library MOFs, adding real structures to the training set
generally improved the real-library 130 K results, especially at 100 bar:

| Training strategy | 130 K / 5 bar MAE | 130 K / 100 bar MAE |
|---|---:|---:|
| Descriptor-only, ToBaCCo only | 1.376 | 4.662 |
| Descriptor-only, ToBaCCo + real | 1.213 | 2.653 |
| Scaled 77 K, ToBaCCo only | 1.317 | 3.567 |
| Scaled 77 K, ToBaCCo + real | 1.130 | 3.222 |
| Fine-tuned GNN, ToBaCCo only | 1.393 | 4.273 |
| **Fine-tuned GNN, ToBaCCo + real** | **0.749** | **2.231** |

The result supports the broader strategy used later for PoroNet: adding
representative real-MOF examples is more valuable than relying only on a
hypothetical-MOF distribution.

### 4.4 Discussion of the 130 K results

The 130 K study demonstrated transferability but also exposed a limitation of
global scaling laws. A simple scale from 77 K can be competitive for 5 bar,
but it cannot consistently reproduce the pressure response at 100 bar. A
fine-tuned representation is more flexible, but the external results show that
the training distribution remains decisive.

The 160 K workflow improved on this design by using direct 160 K simulation
features (`logkH`, `qst0`) rather than relying only on temperature scaling.

---

## 5. Alternative 160 K model studies

### 5.1 Langmuir/physics-informed head

The project tested a physics-informed Langmuir-style head and variants using
different numbers of 160 K labels. The main CoRE test results for the
Langmuir-style PI head were:

| Widom-label count | MAE | Bias | Spearman |
|---:|---:|---:|---:|
| 0 | 0.879 | +0.042 | 0.637 |
| 25 | 0.909 | +0.044 | 0.643 |
| 50 | 0.898 | +0.031 | 0.644 |

With the larger bulk-feature setup:

| Widom-label count | MAE | Bias | Spearman |
|---:|---:|---:|---:|
| 0 | 0.902 | +0.007 | 0.642 |
| 100 | 0.870 | -0.004 | 0.693 |
| 482 | 0.775 | -0.064 | 0.709 |

These results are better than a naive Henry estimate in MAE for some settings,
but they remain well behind the direct `finetune+W` model. The likely reason is
that the PI head imposes a restrictive functional form, while the actual
relationship between structural descriptors, adsorption energy, and uptake is
more complex across real MOFs.

### 5.2 Practical conclusion

The physics-informed head is useful as a diagnostic and potentially as a
regularizer, but it should not replace the empirically validated
fine-tuned GNN-plus-Widom model for production ranking.

---

## 6. PoroNet domain shift and 77 K adaptation

### 6.1 Baseline result

The structure-only 160 K model remained reliable on PoroNet:

```text
MAE:      0.252 g/L
RMSE:     0.472 g/L
Bias:    -0.118 g/L
Spearman: 0.932
```

The baseline hMOF 77 K model was much less reliable:

```text
MAE:      4.599 g/L
RMSE:     5.435 g/L
Bias:    +3.668 g/L
Spearman: 0.809
```

The resulting DOE error was:

```text
MAE:      4.613 g/L
RMSE:     5.509 g/L
Bias:    +3.785 g/L
Spearman: 0.805
```

The positive bias means that the model predicted too much cold-state uptake.
Because DOE capacity subtracts the warm-state uptake, this bias also corrupted
the DOE ranking.

### 6.2 Failure region

The largest errors were associated with:

- low density;
- high void fraction;
- large LCD;
- large PLD;
- high surface area;
- large graph/atom count.

Approximately 50.8% of the evaluated PoroNet structures were outside the hMOF
descriptor domain by at least three standard deviations in one or more
monitored descriptors.

This is a classic extrapolation failure. The hMOF model learned relationships
from a distribution that did not sufficiently cover the low-density,
high-void, large-pore region represented by PoroNet/Tobacco.

### 6.3 Fine-tuning design

The adaptation used:

- 826 training structures;
- 207 held-out structures;
- an 80/20 split;
- structural-difficulty stratification;
- the same graph and descriptor representation;
- direct fine-tuning of the 77 K model.

The split was not designed merely to match the target histogram. It was
designed to preserve difficult structural regions in both train and test.

### 6.4 Held-out result

| Model | Target | MAE | RMSE | Bias | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | 77 K / 100 bar | 4.380 | 5.125 | +3.547 | 0.775 |
| **Fine-tuned** | **77 K / 100 bar** | **0.601** | **1.161** | **+0.107** | **0.986** |
| Baseline | DOE deliverable | 6.657 | 9.168 | -3.165 | 0.053 |
| **Fine-tuned** | **DOE deliverable** | **0.586** | **1.120** | **+0.102** | **0.983** |

The fine-tuned model reduces the 77 K MAE by approximately 86%:

```text
4.380 -> 0.601 g/L
```

It also changes DOE rank correlation from essentially unusable to highly useful:

```text
0.053 -> 0.983 Spearman
```

### 6.5 Discussion

The improvement indicates that the original model was not fundamentally unable
to represent the relevant physics. Instead, it had learned a biased mapping for
an underrepresented region. A relatively modest number of representative
paired labels corrected the local mapping.

This result supports active, region-aware fine-tuning rather than indiscriminate
random expansion of the training set. The next fine-tuning examples should be
selected from regions where:

- the baseline residual is large;
- the descriptor distance from the training domain is large;
- ensemble disagreement is high;
- the candidate could materially affect the top-ranked set.

---

## 7. Ranking, top-candidate screening, and false positives

### 7.1 Why ranking quality matters

A model can have a reasonable global correlation and still fail as a screening
tool if it misorders the top candidates. The project therefore compared
predicted ranks with GCMC-validated ranks and explicitly inspected false
positives.

The original PoroNet ranking showed:

- predicted top-100 structures with true ranks around 500--560;
- true high-performing structures placed below approximately ranks 200--400;
- systematic overprediction in the low-density/high-void region.

This confirms that pointwise MAE alone is insufficient for screening.

### 7.2 CoRE top-10 validation

For the initial CoRE top-10 validation set, the averaged errors were:

| Quantity | MAE | Bias | RMSE |
|---|---:|---:|---:|
| 77 K / 100 bar prediction | 3.602 | +3.602 | 4.129 |
| 160 K / 5 bar prediction | 0.051 | +0.006 | 0.060 |
| DOE prediction | 3.625 | +3.625 | 4.134 |

The warm-state model was accurate, while the cold-state model dominated the
screening error. This is consistent with the PoroNet diagnosis.

### 7.3 Fine-tuned top-20 validation

On a 20-candidate fine-tuned screening validation:

| DOE estimator | MAE | Bias | RMSE |
|---|---:|---:|---:|
| Original DOE estimate | 4.289 | -3.968 | 5.249 |
| Fine-tuned DOE estimate | **1.695** | **+1.502** | **2.292** |

The adapted model substantially reduced the ranking error, although the
remaining positive bias shows that targeted GCMC validation is still required.

### 7.4 Bottom-candidate control

On a five-candidate low-ranked control set:

| DOE estimator | MAE | Bias | RMSE |
|---|---:|---:|---:|
| Original DOE estimate | 3.879 | +1.152 | 4.371 |
| Fine-tuned DOE estimate | 2.994 | -2.994 | 3.637 |

The improvement is smaller than for the top-20 set. This is expected: the
adapted model was optimized to correct the main domain-shift region, not to
perfectly model every low-performing structure.

### 7.5 Screening policy

The preferred screening score is not simply predicted DOE. Candidate selection
should combine:

1. high predicted DOE;
2. low ensemble uncertainty;
3. in-domain descriptor values;
4. successful pore analysis;
5. reasonable graph size;
6. chemical/topological diversity.

High predicted DOE combined with high uncertainty or strong domain violation
should trigger additional RASPA simulations rather than immediate promotion.

---

## 8. RASPA and GCMC validation

### 8.1 Standard force-field convention

The production convention is `UFF_FHm2`:

- UFF framework interactions;
- single-site H2;
- `epsilon/k = 36.7 K`;
- `sigma = 2.958 A`;
- Feynman-Hibbs correction;
- H2 mass `2.016 u`;
- no framework charges;
- 12.8 A cutoff.

### 8.2 One-sample reproduction

`C19_BADJIC_clean` was independently simulated at 160 K / 5 bar. Two runs
gave approximately 2.432 and 2.596 g/L against an archived value of 2.526 g/L.
The spread is consistent with stochastic GCMC behavior.

### 8.3 hMOF/CoRE GCMC checks

The repository contains broader GCMC validation tables. Their main lesson is
that several archived hMOF labels are not reproduced uniformly by an
independent force-field run. The validation set includes suspected label-error
controls, so aggregate error must not be interpreted as a pure model error.

For the 14-row `gcmc_validation.csv` comparison:

| Reference quantity | MAE against GCMC | Bias | RMSE |
|---|---:|---:|---:|
| 77 K / 100 bar label | 7.809 | -0.915 | 11.856 |
| Reference deliverable capacity | 6.741 | -2.030 | 11.541 |

The large RMSE is driven by known anomalous or suspected-label-error cases.
Normal control structures should be evaluated separately from those cases.

### 8.4 Topology validation

For 12 generated-topology structures:

| Prediction | MAE vs first GCMC | Bias | RMSE |
|---|---:|---:|---:|
| 77 K / 100 bar | 7.405 | +7.405 | 7.778 |
| 77 K / 2 bar | 2.977 | +2.977 | 3.467 |
| DOE capacity | 4.501 | +4.405 | 5.317 |

An alternate simulation replicate improved the 100 bar and DOE discrepancies:

| Prediction | MAE vs second GCMC | Bias | RMSE |
|---|---:|---:|---:|
| 77 K / 100 bar | 4.692 | +4.692 | 5.109 |
| DOE capacity | 2.379 | +1.693 | 2.849 |

The result again shows that pressure-state and force-field/simulation
conventions can have a large effect on apparent model error.

---

## 9. Force-field and hydrogen-convention sensitivity

The project tested alternative hydrogen conventions to determine whether
simulation setup could explain discrepancies.

For the convention comparison:

| Convention | Mean absolute difference in 77 K / 100 bar uptake vs reference |
|---|---:|
| `UFF_FH` | 4.387 g/L |
| Classical/no H2-H2 FH | 3.571 g/L |
| `UFF_FHm2` mass adjustment | lower difference in DOE endpoint comparison |

For the DOE comparison between the reference endpoint and the mass-adjusted
variant, the mean absolute difference was approximately 0.886 g/L.

The practical conclusion is that the hydrogen mass and Feynman-Hibbs
convention are not cosmetic details. The exact force-field convention must be
recorded with every label table and kept fixed between training, validation,
and production screening.

---

## 10. Zeo++ and descriptor validation

Zeo++ is used for LCD, PLD, surface area, and void-fraction features. The
project compared fast and accurate pore calculations. The main operational
finding is:

- fast calculations are useful for triage and candidate selection;
- accurate calculations should be used for final labels and validation;
- high-porosity structures are especially sensitive to descriptor differences;
- failed or zero surface-area calculations must be flagged rather than
  silently interpreted as physical values.

Descriptor sensitivity on eight high-performing CoRE candidates showed that
small changes in void fraction and density can produce meaningful uptake
changes. For a `+0.05` void-fraction perturbation, the predicted 77 K uptake
change in the sensitivity table ranged approximately from 1.22 to 2.75 g/L.
For a 10% density perturbation, the corresponding change ranged approximately
from 0.07 to 1.51 g/L.

This explains why descriptor consistency is important for both model training
and applicability-domain detection.

---

## 11. Experimental reference-material predictions

The project also generated baseline and fine-tuned 77 K predictions for four
well-known reference materials:

| Material | Original 77 K / 2 bar | Original 77 K / 100 bar | Fine-tuned 77 K / 2 bar | Fine-tuned 77 K / 100 bar |
|---|---:|---:|---:|---:|
| IRMOF-1 | 9.713 | 52.853 | 10.549 | 53.745 |
| UiO-66 | 22.007 | 36.251 | 21.344 | 37.629 |
| Zn-DOBDC | 21.917 | 43.747 | 22.052 | 46.950 |
| Cu-BTC | 18.812 | 48.503 | 21.167 | 52.010 |

These values are screening predictions, not experimental reproductions. They
are useful for checking that the fine-tuned model produces plausible outputs
for recognizable framework families.

---

## 12. What the results mean scientifically

### 12.1 Geometry alone is not enough

Density, void fraction, LCD, PLD, and surface area explain broad uptake trends,
but they do not fully encode adsorption energetics. The improvement from
Widom features shows that direct low-loading gas-framework information is
valuable for predicting the 160 K endpoint.

### 12.2 Temperature transfer is asymmetric

The hMOF representation transfers more easily to 160 K when Widom features are
available than it transfers to 77 K on a structurally different database.
This is not contradictory. The 160 K model receives target-specific
thermodynamic features, while the baseline 77 K model is asked to extrapolate
from the hMOF distribution without equivalent correction data.

### 12.3 DOE capacity amplifies endpoint errors

DOE capacity is a difference of two predictions. If the cold-state model has a
positive bias, the DOE estimate is also biased high. If the warm-state model
has a positive bias, the DOE estimate is biased low. The two errors can
occasionally cancel, but this should not be relied on.

### 12.4 Ranking requires domain awareness

A candidate with the highest point prediction is not necessarily the best
candidate for simulation. The most useful production output is a ranked table
containing prediction, uncertainty, descriptor distance, atom count, and
whether Widom features were used.

### 12.5 Fine-tuning should be targeted

The PoroNet result shows that selecting additional training structures from
high-error and out-of-domain regions is more effective than adding arbitrary
random structures. This is a practical active-learning strategy for future
datasets.

---

## 13. Limitations and uncertainty

### 13.1 Simulation uncertainty

GCMC outputs are stochastic and finite-cycle. Differences of a few percent can
occur between independent runs. Model errors should be compared with the
simulation block uncertainty.

### 13.2 Label and convention uncertainty

Some archived labels appear inconsistent with independent RASPA runs. Suspected
label-error cases must be isolated before using them to judge model quality.

### 13.3 External sample sizes

The external CoRE/hMOF set contains only 11 structures. Its agreement with the
main test result is encouraging, but confidence intervals are wide.

### 13.4 PoroNet coverage

Only 1,033 of 1,990 PoroNet structures were featurized under the default
600-atom guard. The guard can be increased with `MOF_MAX_ATOMS`, but the
fine-tuning result reported here applies directly to the accepted subset, not
automatically to the 957 rejected structures. Larger graphs should be processed
incrementally because graph memory, neighbor construction, and Zeo++ runtime
increase with structure size.

### 13.5 No direct QMOF DOE validation

QMOF-GCMC metadata was found to emphasize working-capacity targets rather than
the exact paired absolute endpoints. It was not merged into the DOE validation
set.

### 13.6 Labels-only 19,329-row database

The uploaded `f1_*` label table cannot be screened until its matching CIF files
or an ID-to-CIF mapping are recovered.

---

## 14. Recommended production interpretation

For CoRE-like structures with 160 K Widom descriptors:

```text
Use: finetune+W 160 K ensemble
```

For PoroNet/Tobacco-like structures:

```text
Use: PoroNet-adapted 77 K ensemble
Use: structure-only 160 K ensemble if Widom data is unavailable
```

For each candidate, report:

- predicted 77 K / 100 bar uptake;
- predicted 160 K / 5 bar uptake;
- predicted DOE capacity;
- ensemble standard deviations;
- maximum descriptor z-score;
- graph atom count;
- whether the structure is inside the applicability domain;
- whether Widom features were available;
- whether independent RASPA validation is recommended.

The model should be used to prioritize simulations and experiments, not to
replace them for final performance claims.

---

## 15. Final conclusion

The project demonstrates a complete and useful workflow for MOF hydrogen
screening:

1. hMOF GNN pretraining provides a transferable structural representation;
2. real-MOF 160 K transfer learning is highly accurate when combined with
   Widom thermodynamic features;
3. the original 77 K model has a measurable and interpretable domain shift;
4. structural-difficulty-stratified fine-tuning corrects that shift on PoroNet;
5. ranking quality improves substantially after adaptation;
6. RASPA validation confirms that the simulation and parsing pipeline works for
   real CoRE structures;
7. force-field conventions and label quality materially affect apparent error;
8. uncertainty and applicability-domain flags are essential for selecting
   candidates for expensive GCMC.

The strongest current production combination is therefore:

```text
77 K endpoint:  PoroNet-adapted 77 K ensemble for Tobacco/PoroNet-like MOFs
160 K endpoint: CoRE finetune+W ensemble when Widom features are available
DOE ranking:    difference of the endpoint predictions with uncertainty/domain flags
Final proof:    targeted RASPA GCMC and, ultimately, experimental validation
```
