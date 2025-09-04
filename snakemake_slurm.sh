#!/bin/bash
#SBATCH --job-name=snakemake_slurm
#SBATCH --time=01:30:00            # Runtime needs to be longer than the time it takes to complete the workflow. Modify accordingly!
#SBATCH --cpus-per-task=8          
#SBATCH --mem=10G                  
#SBATCH --output=slurm-logs/snakemake_slurm.out
#SBATCH --error=slurm-logs/snakemake_slurm.err

module load mamba
source activate env/

snakemake -s workflow/Snakefile --use-conda --conda-frontend mamba \
  --cores "${SLURM_CPUS_PER_TASK:-8}" "$@"
