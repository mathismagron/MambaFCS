#!/bin/bash
#SBATCH --job-name=MFCS_flops
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=cc-debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=flops-%j.out

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 scipy-stack opencv
source ~/env_mambafcs/bin/activate

cd ~/MambaFCS
python analyze/count_flops.py \
  configs/study/second_4stage.yaml \
  configs/study/second_3stage.yaml \
  configs/study/hiucd_4stage.yaml \
  configs/study/hiucd_3stage.yaml \
  --crop 512 --json flops.json
