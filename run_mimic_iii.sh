#!/usr/bin/env bash
#SBATCH --job-name=GraFITi_MIMIC_III
#SBATCH --output=logs/GraFITi_MIMIC_III_%j.log
#SBATCH --error=err/GraFITi_MIMIC_III_%j.err
#SBATCH --mail-user=shameem@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

cd ~/pyomnits/PyOmniTS
source activate pyomnits

echo "Running dataset: MIMIC_III"
srun sh scripts/GraFITi_periodic/MIMIC_III.sh
# srun python run_test.py