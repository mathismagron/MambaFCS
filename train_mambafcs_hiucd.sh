#!/bin/bash
#SBATCH --job-name=Train_HiUCD
#SBATCH --gres=gpu:h100:1              
#SBATCH --cpus-per-task=8              
#SBATCH --mem=128000M                   
#SBATCH --time=12:00:00                 
#SBATCH --output=train-hiucd-%j.out

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 scipy-stack opencv
source ~/env_mambafcs/bin/activate

echo "Transfert de l'archive vers le nœud de calcul..."
cp $SCRATCH/Hi-UCD_formatted.zip $SLURM_TMPDIR/

echo "Décompression en cours..."
unzip -q $SLURM_TMPDIR/Hi-UCD_formatted.zip -d $SLURM_TMPDIR/

# Génération des listes de fichiers
ls -1 $SLURM_TMPDIR/Hi-UCD_formatted/train/T1 > $SLURM_TMPDIR/Hi-UCD_formatted/train.txt
ls -1 $SLURM_TMPDIR/Hi-UCD_formatted/val/T1 > $SLURM_TMPDIR/Hi-UCD_formatted/val.txt

cd ~/MambaFCS
echo "Démarrage de l'entraînement sur le dataset Hi-UCD..."
python train.py --config configs/train_HIUCD.yaml

echo "Entraînement terminé !"
