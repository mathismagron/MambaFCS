"""Mesure paramètres + FLOPs de STMambaSCD pour une ou plusieurs configs d'entraînement.

Sert à quantifier le gain de l'ablation (baseline 4 stages vs 3 stages).
À lancer sur un nœud GPU (les FLOPs nécessitent un forward des noyaux Mamba CUDA) :

    python analyze/count_flops.py \
        configs/study/second_4stage.yaml \
        configs/study/second_3stage.yaml \
        --crop 512

Les FLOPs des scans SSM peuvent être sous-comptés par fvcore (opérateurs custom) :
le nombre de PARAMÈTRES est, lui, exact et suffit à démontrer la réduction du modèle.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import yaml


def _setup_paths():
    here = os.path.dirname(os.path.abspath(__file__))          # .../MambaFCS/analyze
    mamba_dir = os.path.dirname(here)                          # .../MambaFCS
    repo_root = os.path.dirname(mamba_dir)                     # parent contenant MambaFCS
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


REPO_ROOT = _setup_paths()

import torch  # noqa: E402
from MambaFCS.changedetection.configs.config import get_config  # noqa: E402
from MambaFCS.changedetection.models.STMambaSCD import STMambaSCD  # noqa: E402


def _resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def build_model(train_cfg_path):
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        tcfg = yaml.safe_load(f)

    args = SimpleNamespace(cfg=_resolve(tcfg["cfg"]), opts=tcfg.get("opts"))
    config = get_config(args)
    num_classes = int(tcfg.get("num_classes", 7))

    model = STMambaSCD(
        output_cd=2,
        output_clf=num_classes,
        pretrained=None,  # pas besoin des poids pour compter FLOPs/params
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
    stages = list(config.MODEL.VSSM.DEPTHS)
    dims = list(model.encoder.dims)
    return model, stages, dims


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, crop):
    """FLOPs approximatifs d'un forward (pre, post). Ne lève jamais : renvoie None
    si le calcul échoue. Les scans SSM (noyaux CUDA custom) restent sous-comptés
    quel que soit l'outil → chiffre à lire comme un ordre de grandeur ; les params
    (exacts) restent la mesure de référence pour la réduction du modèle."""
    if not torch.cuda.is_available():
        print("  (pas de GPU : FLOPs non calculés, noyaux Mamba requis)")
        return None
    model = model.cuda().eval()
    pre = torch.randn(1, 3, crop, crop, device="cuda")
    post = torch.randn(1, 3, crop, crop, device="cuda")

    # 1) Compteur natif PyTorch : robuste (gère les einsum à >2 opérandes, contrairement
    #    à fvcore qui plante dessus — AssertionError dans einsum_flop_jit).
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fcm = FlopCounterMode(display=False)
        with torch.no_grad(), fcm:
            model(pre, post)
        total = fcm.get_total_flops()
        if total and total > 0:
            return total
    except Exception as e:
        print(f"  (FlopCounterMode indisponible : {e})")

    # 2) Repli fvcore (peut échouer sur les opérateurs Mamba : on ne plante pas).
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fa = FlopCountAnalysis(model, (pre, post))
            fa.unsupported_ops_warnings(False)
            fa.uncalled_modules_warnings(False)
            total = fa.total()
        return total
    except Exception as e:
        print(f"  (FLOPs non calculables (fvcore) : {e}) — on garde les params")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="Configs d'entraînement (yaml) à mesurer.")
    ap.add_argument("--crop", type=int, default=512, help="Taille d'entrée pour les FLOPs.")
    ap.add_argument("--json", default=None,
                    help="Écrit params/FLOPs dans ce JSON (clé = nom de la config), "
                         "consommable par analyze/aggregate_metrics.py --flops-json.")
    args = ap.parse_args()

    rows = []
    flops_map = {}
    for cfg_path in args.configs:
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        print(f"\n=== {name} ===")
        model, stages, dims = build_model(cfg_path)
        params = count_params(model)
        print(f"  stages (depths) : {stages}")
        print(f"  dims encodeur   : {dims}")
        print(f"  paramètres      : {params / 1e6:.2f} M")
        try:
            flops = count_flops(model, args.crop)
        except Exception as e:  # sécurité : ne jamais interrompre la boucle
            print(f"  (FLOPs ignorés : {e})")
            flops = None
        if flops is not None:
            print(f"  FLOPs @ {args.crop}² : {flops / 1e9:.2f} G")
        rows.append((name, params, flops))
        flops_map[name] = {
            "params_m": round(params / 1e6, 3),
            "gflops": round(flops / 1e9, 3) if flops is not None else None,
        }

    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(flops_map, f, indent=2, ensure_ascii=False)
        print(f"\nÉcrit : {args.json}")

    if len(rows) >= 2:
        print("\n=== Comparaison (référence = 1re config) ===")
        base_p, base_f = rows[0][1], rows[0][2]
        for name, p, f in rows:
            # signe intuitif : négatif = réduction par rapport à la baseline (1re config).
            dp = 100.0 * (p / base_p - 1) if base_p else 0.0
            line = f"  {name:<24} params {p/1e6:7.2f} M ({dp:+.1f}%)"
            if f is not None and base_f:
                df = 100.0 * (f / base_f - 1)
                line += f" | FLOPs {f/1e9:7.2f} G ({df:+.1f}%)"
            print(line)


if __name__ == "__main__":
    main()
