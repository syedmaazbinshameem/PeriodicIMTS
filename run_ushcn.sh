#!/usr/bin/env bash
#SBATCH --job-name=GraFITi_USHCN
#SBATCH --output=logs/GraFITi_USHCN_%j.log
#SBATCH --error=err/GraFITi_USHCN_%j.err
#SBATCH --mail-user=shameem@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

cd ~/pyomnits/PyOmniTS
source activate pyomnits

echo "Running dataset: USHCN"
srun sh scripts/GraFITi_periodic/USHCN.sh
# srun python run_test.py