#!/usr/bin/env bash
#SBATCH --job-name=GraFITi_P12
#SBATCH --output=logs/GraFITi_P12_%j.log
#SBATCH --error=err/GraFITi_P12_%j.err
#SBATCH --mail-user=shameem@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

cd ~/pyomnits/PyOmniTS
source activate pyomnits

echo "Running dataset: P12"
srun sh scripts/GraFITi_periodic/P12.sh
# srun python run_test.py