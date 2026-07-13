"""Génère les prédictions de soumission pour le benchmark en ligne Hi-UCD.

Format attendu par la plateforme
--------------------------------
Un PNG **3 canaux** par paire d'images (512x512) :
    canal 0 = classe sémantique T1
    canal 1 = classe sémantique T2
    canal 2 = changement
avec les indices **réduits de 1** (la plateforme exclut la classe « unlabeled ») :

  * Sémantique — labels : 0=unlabeled, 1=Water … 9=woodland.
    Le modèle sort 10 canaux mais la classe 0 n'est jamais supervisée
    (`label==0 -> 255` à l'entraînement). On fait donc l'argmax **sur les canaux 1..9**
    puis on retire 1 : `argmax(out[:, 1:])` donne directement **0..8**. Ça garantit
    aussi qu'on ne prédit jamais « unlabeled » (qui serait un indice -1 invalide).

  * Changement — labels : 0=unlabeled, 1=unchanged, 2=changed → réduits de 1 :
    0=unchanged, 1=changed. Le modèle sort déjà 2 canaux (0=inchangé, 1=changé),
    donc `argmax(out_cd)` est **déjà au bon format**, aucun décalage à appliquer.

Exemple
-------
    python changedetection/script/predict_hiucd_submission.py \
        --config configs/train_HIUCD.yaml \
        --checkpoint saved_models/HiUCD_FULL/270000_model_0.043.pth \
        --t1-dir $SLURM_TMPDIR/Hi-UCD_formatted/test/T1 \
        --t2-dir $SLURM_TMPDIR/Hi-UCD_formatted/test/T2 \
        --out-dir $SLURM_TMPDIR/submission_hiucd \
        --zip submission_hiucd.zip
"""

import argparse
import os
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))            # .../MambaFCS/changedetection/script
_MAMBA = os.path.dirname(os.path.dirname(_HERE))              # .../MambaFCS
_ROOT = os.path.dirname(_MAMBA)                               # dossier parent de MambaFCS
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from MambaFCS.changedetection.configs.config import get_config  # noqa: E402
from MambaFCS.changedetection.models.STMambaSCD import STMambaSCD  # noqa: E402
import MambaFCS.changedetection.datasets.imutils as imutils  # noqa: E402


def _resolve(path):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(_ROOT, path))


def build_model(train_cfg_path):
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        tcfg = yaml.safe_load(f)
    config = get_config(SimpleNamespace(cfg=_resolve(tcfg["cfg"]), opts=tcfg.get("opts")))
    num_classes = int(tcfg.get("num_classes", 10))

    model = STMambaSCD(
        output_cd=2,
        output_clf=num_classes,
        pretrained=None,  # les poids viennent du checkpoint
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )
    return model, num_classes


def load_pair(t1_path, t2_path):
    """Même prétraitement qu'à l'entraînement : normalisation ImageNet puis CHW."""
    pre = np.array(Image.open(t1_path).convert("RGB"), np.float32)
    post = np.array(Image.open(t2_path).convert("RGB"), np.float32)
    pre = np.transpose(imutils.normalize_img(pre), (2, 0, 1))
    post = np.transpose(imutils.normalize_img(post), (2, 0, 1))
    return pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML d'entraînement (donne l'archi + num_classes).")
    ap.add_argument("--checkpoint", required=True, help="Poids du modèle (state_dict).")
    ap.add_argument("--t1-dir", required=True, help="Images T1 (2018) du test set.")
    ap.add_argument("--t2-dir", required=True, help="Images T2 (2019) du test set.")
    ap.add_argument("--out-dir", required=True, help="Dossier de sortie des PNG de soumission.")
    ap.add_argument("--zip", default=None, help="Si fourni, zippe les PNG dans ce fichier.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="N'traiter que les N premières paires (test rapide).")
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.t1_dir) if n.lower().endswith(".png"))
    if args.limit:
        names = names[: args.limit]
    if not names:
        ap.error(f"Aucun PNG trouvé dans {args.t1_dir}")
    print(f"{len(names)} paires à prédire")

    model, num_classes = build_model(args.config)
    state = torch.load(_resolve(args.checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model" in state:   # tolère un checkpoint complet
        state = state["model"]
    model.load_state_dict(state)
    model = model.cuda().eval()
    print(f"Checkpoint chargé : {args.checkpoint}  (num_classes={num_classes})")

    os.makedirs(args.out_dir, exist_ok=True)

    written = 0
    with torch.no_grad():
        for start in range(0, len(names), args.batch_size):
            chunk = names[start: start + args.batch_size]
            pres, posts = [], []
            for n in chunk:
                pre, post = load_pair(os.path.join(args.t1_dir, n), os.path.join(args.t2_dir, n))
                pres.append(pre)
                posts.append(post)
            pre_t = torch.from_numpy(np.stack(pres)).cuda()
            post_t = torch.from_numpy(np.stack(posts)).cuda()

            out_cd, out_t1, out_t2 = model(pre_t, post_t)

            # indices déjà "réduits de 1" (cf. docstring)
            chg = out_cd.argmax(dim=1).to(torch.uint8).cpu().numpy()          # 0..1
            sem1 = out_t1[:, 1:].argmax(dim=1).to(torch.uint8).cpu().numpy()  # 0..8
            sem2 = out_t2[:, 1:].argmax(dim=1).to(torch.uint8).cpu().numpy()  # 0..8

            for i, n in enumerate(chunk):
                rgb = np.stack([sem1[i], sem2[i], chg[i]], axis=-1)  # H x W x 3
                Image.fromarray(rgb, mode="RGB").save(os.path.join(args.out_dir, n))
                written += 1

            if start % (args.batch_size * 100) == 0:
                print(f"  {written}/{len(names)}", flush=True)

    print(f"{written} PNG écrits dans {args.out_dir}")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for n in names:
                zf.write(os.path.join(args.out_dir, n), arcname=n)  # à plat, pas de dossier
        size_mb = os.path.getsize(args.zip) / 1e6
        print(f"Archive prête : {args.zip} ({size_mb:.1f} Mo, {len(names)} masques)")


if __name__ == "__main__":
    main()
