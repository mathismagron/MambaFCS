#!/bin/bash
#SBATCH --job-name=MFCS_HiUCD_3stage
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256000M
#SBATCH --time=72:00:00
#SBATCH --output=train-hiucd-3stage-%j.out

# [ÉTUDE ablation encodeur] Hi-UCD — 3 stages (stage 1/32 retiré), départ ImageNet.
# Reprise automatique en cas d'annulation TIME LIMIT : resoumettre ce même script
# (l'entraînement repart depuis saved_models/study/HiUCD_3stage/last_checkpoint.pth).

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 scipy-stack opencv
source ~/env_mambafcs/bin/activate

echo "Transfert de l'archive Hi-UCD vers le nœud de calcul..."
cp $SCRATCH/Hi-UCD_formatted.zip $SLURM_TMPDIR/

echo "Décompression en cours..."
unzip -q $SLURM_TMPDIR/Hi-UCD_formatted.zip -d $SLURM_TMPDIR/

# Génération des listes de fichiers
ls -1 $SLURM_TMPDIR/Hi-UCD_formatted/train/T1 > $SLURM_TMPDIR/Hi-UCD_formatted/train.txt
ls -1 $SLURM_TMPDIR/Hi-UCD_formatted/val/T1   > $SLURM_TMPDIR/Hi-UCD_formatted/val.txt

cd ~/MambaFCS
echo "Démarrage de l'entraînement Hi-UCD (3 stages, depuis ImageNet)..."
python train.py --config configs/study/hiucd_3stage.yaml

echo "Entraînement terminé !"
