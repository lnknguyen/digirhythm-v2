# DigiRhythm

Quantifying day‑to‑day routine structure from the human's digital traces people.
This repository contains a reproducible Snakemake pipeline with example configurations and a Conda environment. Experiments currently target three studies (`Tesserae`, `MoMo-Mood`, `GLOBEM`) and can be extended to additional cohorts with minimal changes.

---

## 1  Project structure

```text
├── config/                 # YAML file that drives every run
│   └── config.yaml
├── workflow/               # Snakemake workflow
│   ├── Snakefile
│   └── rules/              # modular rule files (pre‑process, cluster, signature…)
├── data/                   # place to drop raw input data (git‑ignored)
├── out/                    # auto‑generated results (clusters, signatures, plots…)
├── slurm-logs/             # job logs when run on an HPC cluster
├── env.yml                 # Conda environment specification
├── snakemake_slurm.sh      # convenience wrapper for SLURM
└── *.png / *.txt           # example output artefacts
```

| File                 | Purpose                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| `config/config.yaml` | Lists studies, features, and clustering settings such as algorithm (`gmm` by default) and number of components. |
| `workflow/Snakefile` | Orchestrates the full pipeline and includes individual rule files under `workflow/rules/`.                      |
| `env.yml`            | Creates a light Conda‑Forge environment (Python 3 + Snakemake, Black, Graphviz).                                |

---

## 2  How to run

### 2.1  Local execution

```bash
# 1. Set up the Conda environment
mamba env create -f env.yml      # or: conda env create -f env.yml
conda activate conda-digirhythm

# 2. Launch the pipeline 
snakemake --snakefile workflow/Snakefile
```

### 2.2  SLURM / Triton cluster

A SLURM wrapper is provided:

```bash
sbatch snakemake_slurm.sh
```

The script loads *mamba*, activates the environment, and executes Snakemake script. Rule‑specific environments are created on demand.

---

## 3  Updating the configuration

All study‑specific options live in **`config/config.yaml`**:

| Key                | Description                                                                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `groupby`          | How raw records are aggregated for each dataset (e.g., `user`, `date`, `wave`).                                                                                 |
| `features`         | Behavioural variables to feed into clustering (screen‑time windows, call durations, sleep metrics, etc.).                                                       |
| `cluster_settings` | Per‑study algorithm (`gmm` by default), data‑splitting policy, inclusion threshold, and whether to run automatic model‑selection (`run_model_selection: true`). |
| `signature`        | Choose whether signatures are ranked or unranked.                                                                                                               |

After editing `config.yaml`, simply re‑run Snakemake; the DAG will detect what needs to be recomputed.
If you add or remove whole studies, also update `STUDIES` in `workflow/rules/common.smk`.
