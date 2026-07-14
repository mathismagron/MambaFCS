#!/bin/bash
#SBATCH --job-name=HiUCD_submit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=3:00:00
#SBATCH --output=submit-hiucd-%j.out

# Génère les PNG de soumission pour le benchmark en ligne Hi-UCD (test set, 21600 paires)
# puis les zippe dans $SCRATCH/submission_hiucd.zip (SLURM_TMPDIR est effacé en fin de job).
#
# Checkpoint utilisé = le meilleur du run "depuis SECOND" (SeK 0.043).

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 scipy-stack opencv
source ~/env_mambafcs/bin/activate

echo "Transfert + décompression du dataset..."
cp $SCRATCH/Hi-UCD_formatted.zip $SLURM_TMPDIR/
unzip -q $SLURM_TMPDIR/Hi-UCD_formatted.zip -d $SLURM_TMPDIR/

echo "--- contenu du split test ---"
ls $SLURM_TMPDIR/Hi-UCD_formatted/test/ 2>/dev/null || echo "!! pas de dossier test/ dans l'archive"
echo -n "nb images test/T1 : "; ls -1 $SLURM_TMPDIR/Hi-UCD_formatted/test/T1 2>/dev/null | wc -l
echo -n "nb images test/T2 : "; ls -1 $SLURM_TMPDIR/Hi-UCD_formatted/test/T2 2>/dev/null | wc -l
echo "-----------------------------"

cd ~/MambaFCS
python changedetection/script/predict_hiucd_submission.py \
  --config configs/train_HIUCD.yaml \
  --checkpoint saved_models/HiUCD_FULL/270000_model_0.043.pth \
  --t1-dir $SLURM_TMPDIR/Hi-UCD_formatted/test/T1 \
  --t2-dir $SLURM_TMPDIR/Hi-UCD_formatted/test/T2 \
  --out-dir $SLURM_TMPDIR/submission_hiucd \
  --zip $SCRATCH/submission_hiucd.zip \
  --batch-size 4 || { echo "!! ÉCHEC de la génération des prédictions"; exit 1; }

echo "Terminé ! Archive prête : $SCRATCH/submission_hiucd.zip"
ls -lh $SCRATCH/submission_hiucd.zip
