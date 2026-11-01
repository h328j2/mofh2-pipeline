# ML screening of MOFs for cryogenic H2 storage — full reproduction package

Everything needed to reproduce the project on Kaggle: scripts, the exact force field, every structure that was simulated,
the selection lists, the hMOF model checkpoints, and the original raw RASPA outputs (`reference/reference_runs.tgz`) for comparison.

## How to use
1. Upload this folder (zipped) as a Kaggle **Dataset** (e.g. name it `mofh2-pipeline`).
2. Import the notebooks `00`–`08` (File → Import notebook). In each: **Add Input → your dataset**, and turn **Internet ON**
   (Settings) — setup clones RASPA2, Zeo++ and CoRE MOF 2019 from GitHub and pip-installs packages. CPU sessions are enough;
   no GPU is needed.
3. Run them in order. Each notebook ends by saving `stageNN_outputs.tgz`; attach earlier outputs to later notebooks
   (or leave `USE_REFERENCE_UPSTREAM = True`, which fills missing upstream results from the original runs so any notebook can run alone).
4. Long queues stop launching new runs after `MAX_HOURS` (default 11). Save the version, attach its output to a new session,
   and rerun: finished runs are skipped. Split a long notebook across sessions or run notebooks in parallel.

| Notebook | Reproduces | Simulations | Time on 4 cores |
|---|---|---|---|
| 00 setup & validation | tool build; 2 reference GCMC runs; checkpoint check | 2 | ~30 min |
| 01 hMOF models & audit | summary (training itself: repo notebook `Hmof h2 modelling.ipynb`) | 0 | minutes |
| 02 convention & CoRE 2024 | report 5.1–5.6 | 60 + 4 box | 4–6 h |
| 03 topologies | report 6.1–6.3 | 36 | ~2 h |
| 04 beyond 77 K / 130 K | report 6.4 | 665 | 5–7 h |
| 05 CoRE 2019 160 K dataset | report 6.5 | 1,657 | 8–11 h |
| 06 screening & 77 K fine-tuning | report 6.6–6.7 | 710 + featurising 7,206 | 10–14 h (2 sessions) |
| 07 experiment comparison | report 5.7 | 28 + 4 He | ~2 h |
| 08 audit & summary | raw-file audit, comparison with original | 0 | minutes |

## Simulation setup (identical in every run)
RASPA2; UFF framework Lennard-Jones; single-site H2 (eps/k = 36.7 K, sigma = 2.958 A); Feynman–Hibbs correction with mass
2.016 u for H2–H2 (the hMOF-matched convention; `UFF_FHm2`); no charges; 12.8 A cutoff; unit cells so every perpendicular
width >= 25.6 A. `UFF_FH` (1.008 u) and `UFF_noFHH2` (classical) are the alternative conventions tested in notebook 02.

## Notes on exact reproduction
* GCMC is stochastic: expect run-to-run differences of ~0.5–3% (block-average errors are in every output file).
* Cycle counts are uniform per queue here (77 K fine-tuning labels: 400 + 400 at 100 bar). In the original run the last 28
  of those 150 used 250 + 250 to save time.
* Selection lists (`core_stageA.csv`, `core_stageB.csv`, `ft77_train.csv`, `t130_picks.csv`, `real_picks.csv`) are the
  original ones, so the same MOFs are simulated. Notebook 05 can redo the selection (`RESELECT = True`).
* Original result tables are under `data/`; `compare_reference.py` prints differences against them.

## Layout
`src/` scripts · `data/` selection lists, queues, original result tables, `raspa_cif/` all simulated structures, `user_zip/` hMOF/CoRE inputs ·
`forcefield/`, `molecules/` RASPA definitions · `model_repo/` hMOF checkpoints · `tobacco/` ToBaCCo 3.0 code · `reference/` original raw outputs ·
`setup_env.sh` builds everything into /kaggle/working/mofh2.
