#!/usr/bin/env bash
#SBATCH --job-name=ETTh1
#SBATCH --output=logs/GraFITi_ETTh1_%j.log
#SBATCH --error=err/GraFITi_ETTh1_%j.err
#SBATCH --mail-user=shameem@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

cd ~/pyomnits/PyOmniTS
source activate pyomnits

echo "Running dataset: ETTh1"
srun sh scripts/GraFITi/ETTh1.sh
# srun python run_test.py