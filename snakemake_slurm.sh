#!/bin/bash
#SBATCH --job-name=snakemake_slurm
#SBATCH --time=01:00:00            # Runtime needs to be longer than the time it takes to complete the workflow. Modify accordingly!
#SBATCH --cpus-per-task=2          
#SBATCH --mem=30G                  
#SBATCH --output=slurm-logs/snakemake_slurm.out
#SBATCH --error=slurm-logs/snakemake_slurm.err

module load mamba
source activate env/
snakemake --snakefile workflow/Snakefile  --software-deployment-method conda --conda-frontend mamba --cores 2
#snakemake --snakefile workflow/Snakefile --cores 2
