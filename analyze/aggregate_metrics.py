"""Agrège les métriques finales des runs (.out SLURM) en un tableau Markdown.

Parse chaque fichier .out, retient la validation au **meilleur SeK** (robuste même si
le run a été coupé sur TIME LIMIT — la ligne « best round » de fin de training n'est
alors pas présente), et produit le tableau de résultats de l'étude d'ablation.

Exemples :

    # labels déduits des noms de fichiers (train-<label>-<jobid>.out)
    python analyze/aggregate_metrics.py train-second-4stage-*.out train-second-3stage-*.out

    # tous les .out d'un dossier + params/FLOPs mesurés au préalable
    python analyze/count_flops.py configs/study/*.yaml --json flops.json
    python analyze/aggregate_metrics.py --dir . --flops-json flops.json

Les colonnes Params/GFLOPs viennent du --flops-json (clé = nom de config, ex.
`second_3stage`) ; le label d'un run est normalisé vers cette même clé.
"""

import argparse
import glob
import json
import os
import re

# "Kappa coefficient rate is <k>, F1 is <f>, OA is <oa>, mIoU is <miou>, SeK is <sek>"
_NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
METRIC_RE = re.compile(
    r"Kappa coefficient rate is " + _NUM +
    r", F1 is " + _NUM +
    r", OA is " + _NUM +
    r", mIoU is " + _NUM +
    r", SeK is " + _NUM
)
ITER_RE = re.compile(r"^iter is (\d+),")
FINISHED_RE = re.compile(r"best round is")
CANCELLED_RE = re.compile(r"CANCELLED|DUE TO TIME LIMIT")


def label_from_path(path):
    """train-second-3stage-17246999.out -> second_3stage ; sinon le stem nettoyé."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"^train-", "", stem)
    stem = re.sub(r"-\d+$", "", stem)  # retire le job id final
    return stem.replace("-", "_")


def parse_out(path):
    best = None  # (sek, kappa, f1, miou, oa, iter)
    last_iter = None
    finished = False
    cancelled = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            mi = ITER_RE.match(line)
            if mi:
                last_iter = int(mi.group(1))
                continue
            mm = METRIC_RE.search(line)
            if mm:
                kappa, f1, oa, miou, sek = (float(x) for x in mm.groups())
                if best is None or sek > best[0]:
                    best = (sek, kappa, f1, miou, oa, last_iter)
                continue
            if FINISHED_RE.search(line):
                finished = True
            if CANCELLED_RE.search(line):
                cancelled = True
    status = "terminé" if finished else ("coupé" if cancelled else "en cours")
    return best, status


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outs", nargs="*", help="Fichiers .out (globs autorisés).")
    ap.add_argument("--dir", default=None, help="Dossier où chercher les *.out.")
    ap.add_argument("--flops-json", default=None, help="JSON params/FLOPs (de count_flops.py).")
    args = ap.parse_args()

    paths = []
    for pat in args.outs:
        paths.extend(sorted(glob.glob(pat)))
    if args.dir:
        paths.extend(sorted(glob.glob(os.path.join(args.dir, "*.out"))))
    # dédoublonne en gardant l'ordre
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    if not paths:
        ap.error("Aucun fichier .out trouvé (donne des chemins ou --dir).")

    flops_map = {}
    if args.flops_json and os.path.isfile(args.flops_json):
        with open(args.flops_json, encoding="utf-8") as f:
            flops_map = json.load(f)

    # Si plusieurs .out portent le même label (relances), garde le meilleur SeK.
    runs = {}
    for p in paths:
        label = label_from_path(p)
        best, status = parse_out(p)
        if best is None:
            print(f"# {os.path.basename(p)} : aucune validation trouvée — ignoré")
            continue
        if label not in runs or best[0] > runs[label][0][0]:
            runs[label] = (best, status, os.path.basename(p))

    header = "| Run | Params (M) | GFLOPs @512 | SeK | mIoU | F1 | OA |"
    sep = "|---|---|---|---|---|---|---|"
    print("\n" + header)
    print(sep)
    for label in sorted(runs):
        (sek, kappa, f1, miou, oa, it), status, fname = runs[label]
        fl = flops_map.get(label, {})
        params = fl.get("params_m")
        gflops = fl.get("gflops")
        params_s = f"{params:.2f}" if isinstance(params, (int, float)) else "—"
        gflops_s = f"{gflops:.2f}" if isinstance(gflops, (int, float)) else "—"
        print(f"| {label} | {params_s} | {gflops_s} | "
              f"{fmt(sek)} | {fmt(miou)} | {fmt(f1)} | {fmt(oa)} |")

    # Détail (itération du meilleur round + statut) hors tableau principal.
    print("\nDétail :")
    for label in sorted(runs):
        (sek, kappa, f1, miou, oa, it), status, fname = runs[label]
        it_s = it if it is not None else "?"
        print(f"  - {label:<16} meilleur SeK={sek:.4f} @ iter {it_s} "
              f"(kappa={kappa:.4f}) — {status} [{fname}]")


if __name__ == "__main__":
    main()
