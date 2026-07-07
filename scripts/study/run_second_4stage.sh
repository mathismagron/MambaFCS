#!/bin/bash
#SBATCH --job-name=MFCS_SECOND_4stage
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256000M
#SBATCH --time=72:00:00
#SBATCH --output=train-second-4stage-%j.out

# [ÉTUDE ablation encodeur] SECOND — BASELINE 4 stages, départ ImageNet.
# Reprise automatique en cas d'annulation TIME LIMIT : resoumettre ce même script
# (l'entraînement repart depuis saved_models/study/SECOND_4stage/last_checkpoint.pth).

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 scipy-stack opencv
source ~/env_mambafcs/bin/activate

echo "Transfert de l'archive SECOND vers le nœud de calcul..."
cp $SCRATCH/SECOND_formatted.zip $SLURM_TMPDIR/

echo "Décompression en cours..."
unzip -q $SLURM_TMPDIR/SECOND_formatted.zip -d $SLURM_TMPDIR/

# Génération des listes de fichiers
ls -1 $SLURM_TMPDIR/SECOND/train/T1 > $SLURM_TMPDIR/SECOND/train.txt
ls -1 $SLURM_TMPDIR/SECOND/test/T1  > $SLURM_TMPDIR/SECOND/test.txt

cd ~/MambaFCS
echo "Démarrage de l'entraînement SECOND (4 stages, depuis ImageNet)..."
python train.py --config configs/study/second_4stage.yaml

echo "Entraînement terminé !"
