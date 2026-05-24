# DigiRhythm

Quantifying day-to-day routine structure from passively collected smartphone and wearable sensor data.
The pipeline clusters daily behavioural feature vectors (screen time, physical activity, sleep) into routine types using a Gaussian Mixture Model, then computes per-person *routine signatures* that summarise how an individual moves between those types over time.

Experiments currently target four cohorts — **Tesserae**, **MoMo-Mood**, **GLOBEM**, **DTU** — and can be extended to new studies by adding a config block and a set of preprocessing scripts.

---

## Setup

```bash
mamba env create -f env.yml      # or: conda env create -f env.yml
conda activate conda-digirhythm
```

The environment (`env.yml`) provides Python 3.12, Snakemake, Black, and Ruff.
Per-rule environments (scikit-learn, R, etc.) live under `workflow/envs/` and are created on demand by Snakemake when `--use-conda` is passed.

---

## Running the pipeline

### Local

```bash
snakemake --snakefile workflow/Snakefile --use-conda --cores 8
```

### SLURM / Triton

```bash
sbatch snakemake_slurm.sh
```

The script requests 64 CPUs and 40 GB of memory, loads mamba, activates the local `env/` environment, and passes `--use-conda` to Snakemake so that rule-specific environments are resolved automatically.

---

## Pipeline stages

```
Raw sensor files
    │
    ▼  workflow/rules/preprocess.smk
data/interim/{study}/{sensor}_4epochs.csv      ← per-sensor 4-epoch daily features
    │
    ▼  workflow/rules/preprocess.smk  (clean_features / rename_and_concatenate)
data/processed/{study}/all_features_clean.csv  ← merged, NaN-filtered feature matrix
    │
    ▼  workflow/rules/cluster.smk
out/clusters/{study}/{k}/gmm_cluster.csv       ← cluster assignment per person-day
out/clusters/{study}/{k}/gmm_cluster_centroids.csv
    │
    ▼  workflow/rules/signature.smk
out/signature/{study}/cluster_{k}/{window}/signature_{rank}_{dist}.csv
out/signature/{study}/cluster_{k}/{window}/signature_d_self_{rank}_{dist}.csv
out/signature/{study}/cluster_{k}/{window}/signature_d_ref_{rank}_{dist}.csv
out/transition_signature/…
```

**Features** extracted for each day are segmented into four time-of-day epochs (night, morning, afternoon, evening) plus an all-day aggregate:

| Modality | Feature columns |
| -------- | --------------- |
| Physical activity (steps) | `activity_{epoch}` |
| Screen time | `screen_{epoch}` |
| Sleep | `sleep_onset`, `sleep_offset`, `sleep_duration` |

Not all sensors are available for every study — DTU, for instance, has only screen and sleep.

**Signature distance metrics** default to Jensen-Shannon divergence (`jsd`), which is appropriate for probability distributions. Cosine similarity (`cosine`) is also supported.

---

## Project layout

```
config/
  config.yml                  # single file that drives the entire pipeline
  hdbscan_parameters.tsv      # alternative clustering parameters (experimental)
workflow/
  Snakefile                   # entry point; includes rule files
  rules/
    common.smk                # wildcard constants and all_outputs()
    preprocess.smk            # sensor extraction and feature cleaning
    cluster.smk               # GMM clustering and model selection
    signature.smk             # routine and transition signatures
  scripts/
    make_data/                # combine.py, clean.py
    clusters/                 # run.py (GMM fit / predict)
    signature/                # signature.py, transition_signature.py
    preprocess/
      tesserae/               # per-sensor extraction scripts
      globem/
      momo/
      dtu/
  envs/
    python_env.yaml           # pandas, scikit-learn, etc.
    latent_env.yaml           # heavier latent-model dependencies
    r_env.yaml                # R packages for statistical analysis
  notebooks/                  # exploratory and reporting notebooks
data/                         # raw and processed data (git-ignored)
out/                          # pipeline outputs (git-ignored)
env.yml                       # top-level Conda environment
snakemake_slurm.sh            # SLURM submission wrapper
```

---

## Configuration reference

All settings live in `config/config.yml`.

| Key | Description |
| --- | ----------- |
| `groupby` | Columns used to identify a person-day (e.g. `[user, date]`). Study-specific. |
| `sensors` | List of sensor modalities to extract per study. |
| `features` | Feature columns fed into clustering. Controls which epoch-level variables are used. |
| `cluster_settings` | Per-study clustering configuration (see below). |
| `signature` | Signature computation settings (see below). |

**`cluster_settings` sub-keys**

| Key | Description |
| --- | ----------- |
| `algorithm` | Clustering algorithm (`gmm`). |
| `split` / `strategy` / `group_col` | Whether and how to split data before fitting (e.g. by `wave` for GLOBEM). |
| `min_threshold` / `max_threshold` | Minimum and maximum days required for a participant to be included. |
| `run_model_selection` | Set to `true` to sweep component counts instead of using `optimal_gmm_settings`. |
| `optimal_gmm_settings` | Fixed GMM hyperparameters used when `run_model_selection` is false. |

**`signature` sub-keys**

| Key | Description |
| --- | ----------- |
| `ranked` | Whether to rank-normalise signatures before computing distances. |
| `dist_method` | Distance metric: `jsd` (default) or `cosine`. |
| `split` / `splits` / `split_col` | Whether to compute signatures on held-out splits. |
| `threshold_days` | Minimum active days for a participant to receive a signature. |

After editing `config.yml`, re-run Snakemake; the DAG determines what needs recomputation.
To add a new study, add entries under `groupby`, `sensors`, `features`, `cluster_settings`, and `signature`, add the study name to `STUDIES` in `workflow/rules/common.smk`, and add preprocessing scripts under `workflow/scripts/preprocess/{study}/`.

---

## Code quality

A git pre-commit hook (`.git/hooks/pre-commit`) runs on every `git commit`:

1. **Black** — reformats all Python files in-place.
2. **Ruff** — lints the reformatted code; the commit is aborted if errors remain.

Both tools are pinned in `env.yml`. To run them manually:

```bash
black .
ruff check .
```

If the hook blocks a commit, fix the reported issues, re-stage the affected files, and commit again.
