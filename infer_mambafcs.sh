#!/bin/bash
#SBATCH --job-name=InferMambaFCS
#SBATCH --gres=gpu:h100:1              
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32000M                   
#SBATCH --time=2:00:00                 
#SBATCH --output=infer-%j.out

# 1. Chargement des modules Compute Canada (INCLUANT OPENCV)
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 scipy-stack opencv

# 2. Activation de votre nouvel environnement MambaFCS
source ~/env_mambafcs/bin/activate

# 3. Transfert du dataset SECOND vers le SSD ultra-rapide temporaire du GPU
echo "Transfert de l'archive SECOND.zip vers le noeud de calcul..."
cp $SCRATCH/SECOND_formatted.zip $SLURM_TMPDIR/

echo "Decompression en cours..."
unzip -q $SLURM_TMPDIR/SECOND_formatted.zip -d $SLURM_TMPDIR/

ls -1 $SLURM_TMPDIR/SECOND/train/T1 > $SLURM_TMPDIR/SECOND/train.txt
ls -1 $SLURM_TMPDIR/SECOND/test/T1 > $SLURM_TMPDIR/SECOND/test.txt

cd ~/MambaFCS

echo "Demarrage de l'evaluation sur le dataset SECOND..."
# Le flag --evaluate va bloquer l'entraînement et forcer la validation directe
python train.py --config configs/train_SECOND.yaml --evaluate

echo "Inference termine !"
