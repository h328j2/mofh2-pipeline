# Predictions for Well-Known MOFs

Predictions generated from the repository CIFs using the hMOF 77 K ensemble, the PoroNet-adapted 77 K ensemble, and the structure-only CoRE 160 K fine-tuned ensemble. The 160 K model uses no Widom features here because no Widom inputs were supplied. Values are volumetric uptake in g/L.

## Coverage

- Requested structures: 20
- Successfully featurized: 12
- Flagged too large or unavailable: 8
- Graph limit: 600 atoms

## Usable predictions

| MOF | Atoms | Density | Void fraction | LCD | PLD | Baseline 77K/100bar | Adapted 77K/100bar | 160K/5bar | DOE baseline | DOE adapted | In domain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| IRMOF-15 | 226 | 0.412 | 0.854 | 10.349 | 8.559 | 55.93 | 55.19 | 1.56 | 54.37 | 53.63 | yes |
| IRMOF-1 | 424 | 0.593 | 0.817 | 14.832 | 8.974 | 52.85 | 51.92 | 1.82 | 51.04 | 50.10 | yes |
| Ni-DOBDC | 162 | 1.194 | 0.589 | 11.355 | 12.157 | 44.38 | 47.24 | 2.78 | 41.60 | 44.46 | yes |
| Co-DOBDC | 162 | 1.181 | 0.577 | 11.414 | 12.280 | 44.27 | 46.95 | 2.62 | 41.65 | 44.33 | yes |
| Zn-DOBDC | 162 | 1.220 | 0.606 | 11.602 | 12.411 | 43.75 | 46.97 | 3.02 | 40.73 | 43.95 | yes |
| Mg-DOBDC | 162 | 0.922 | 0.592 | 11.592 | 12.392 | 40.92 | 46.67 | 3.24 | 37.68 | 43.43 | yes |
| ZIF-80 | 456 | 1.076 | 0.578 | 12.036 | 11.679 | 42.23 | 44.08 | 1.68 | 40.55 | 42.41 | yes |
| ZIF-8 | 276 | 0.925 | 0.577 | 11.282 | 3.823 | 41.85 | 41.04 | 1.61 | 40.24 | 39.43 | yes |
| ZIF-68 | 600 | 1.033 | 0.543 | 10.772 | 9.104 | 38.44 | 39.65 | 2.10 | 36.34 | 37.55 | yes |
| MIL-53(Cr)ht | 76 | 1.042 | 0.534 | 7.197 | 7.927 | 38.39 | 40.50 | 3.15 | 35.24 | 37.35 | yes |
| UIO-66 | 432 | 1.219 | 0.519 | 8.452 | 4.260 | 36.25 | 40.06 | 3.11 | 33.14 | 36.95 | yes |
| MIL-53(Cr)lt | 76 | 1.529 | 0.262 | 3.245 | 2.731 | 15.08 | 19.83 | 0.13 | 14.95 | 19.70 | yes |

## Uncertainty and unavailable structures

| MOF | Status | Atoms |
|---|---|---:|
| IRMOF-5 | too_large |  |
| IRMOF-10 | too_large |  |
| IRMOF-13 | too_large |  |
| IRMOF-14 | too_large |  |
| Cu-BTC | too_large |  |
| MOF-210 | too_large |  |
| PCN-60 | too_large |  |
| PCN-61 | too_large |  |

## Interpretation

The baseline 77 K column is the original hMOF ensemble and is the safer reference for familiar hMOFs. The adapted 77 K column is included to show the effect of the PoroNet/Tobacco domain adaptation; it should be preferred only when the target structure family is similar to that adaptation domain. The 160 K predictions are structure-only and therefore less accurate than the Widom-enabled `finetune+W` model used in the CoRE validation. DOE values are endpoint differences, not direct GCMC results.

The top adapted-DOE predictions among the accepted structures are IRMOF-15, IRMOF-1, Ni-DOBDC, Co-DOBDC, Zn-DOBDC, and Mg-DOBDC. This is a model ranking for prioritization, not experimental confirmation. IRMOF-5, IRMOF-10, IRMOF-13, IRMOF-14, Cu-BTC, MOF-210, PCN-60, and PCN-61 exceeded the 600-atom graph limit and require primitive-cell or hierarchical featurization before prediction.
