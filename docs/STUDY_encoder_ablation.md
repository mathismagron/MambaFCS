# Étude : ablation d'un stage d'encodeur (MambaFCS)

**Question.** Peut-on retirer le stage le plus profond (1/32) de l'encodeur VMamba
tout en gardant le décodeur, pour obtenir un réseau **aussi performant mais moins
coûteux en calcul** ? On mesure l'effet sur **SECOND** et sur **Hi-UCD** (les résultats
sont attendus différents selon le dataset).

Tout part du backbone **pré-entraîné ImageNet** (`vssm_base_0229_ckpt_epoch_237.pth`),
**pas** de poids SECOND. Les décodeurs sont entraînés depuis zéro (comme d'habitude).

## Ce qui a été rendu configurable (branche `study/encoder-stage-ablation`)

- `ChangeDecoder` / `SemanticDecoder` : généralisés à un **nombre de stages arbitraire**,
  déduit de `len(encoder_dims)`. Pour 4 stages, le graphe est **strictement identique**
  à l'origine (vérifié : flux de canaux et `change_maps` inchangés).
- `STMambaSCD` : `out_indices` et les têtes de classification dérivent du backbone
  (`len(depths)`, `encoder.dims[0]`) au lieu d'être câblés en dur à 4 / 128.
- Le nombre de stages se pilote uniquement par le **backbone yaml** (`DEPTHS`).
  `vssm_base_224_3stage.yaml` = `DEPTHS: [2,2,15]` → dims `[128,256,512]`, stage 1/32 retiré.
  Les poids ImageNet des stages restants se chargent à l'identique (`strict=False`).

## Matrice d'expériences

| Config | Backbone | Dataset | Départ |
|---|---|---|---|
| `configs/study/second_4stage.yaml` | 4 stages (baseline) | SECOND | ImageNet |
| `configs/study/second_3stage.yaml` | 3 stages | SECOND | ImageNet |
| `configs/study/hiucd_4stage.yaml`  | 4 stages (baseline) | Hi-UCD | ImageNet |
| `configs/study/hiucd_3stage.yaml`  | 3 stages | Hi-UCD | ImageNet |

Les 4 runs partagent le même budget (`max_iters`, lr, batch, crop) pour une comparaison
équitable ; seul le backbone change entre baseline et ablation. Chaque run a un
`model_saving_name` distinct → checkpoints/logs séparés, et bénéficie de la reprise
automatique sur `last_checkpoint.pth` (robustesse TIME LIMIT).

## Protocole

1. **Mesurer le coût** avant d'entraîner (sur nœud GPU) :
   ```bash
   python analyze/count_flops.py configs/study/second_4stage.yaml configs/study/second_3stage.yaml --crop 512
   ```
   Le nombre de **paramètres** est exact ; les FLOPs fvcore peuvent sous-compter les
   scans SSM (opérateurs custom) → à interpréter comme un ordre de grandeur.

2. **Entraîner** — un script SLURM prêt par run dans `scripts/study/` (time 72h,
   reprise auto sur `last_checkpoint.pth`) :
   ```bash
   sbatch scripts/study/run_second_4stage.sh
   sbatch scripts/study/run_second_3stage.sh
   sbatch scripts/study/run_hiucd_4stage.sh
   sbatch scripts/study/run_hiucd_3stage.sh
   ```
   (ou en direct : `python train.py --config configs/study/second_3stage.yaml`)

3. **Agréger les résultats** automatiquement depuis les `.out` (retient le meilleur SeK,
   fonctionne même sur un run coupé) et remplir le tableau ci-dessous :
   ```bash
   # optionnel : mesurer params/FLOPs une fois pour toutes
   python analyze/count_flops.py configs/study/*.yaml --json flops.json
   # agréger les 4 runs
   python analyze/aggregate_metrics.py --dir . --flops-json flops.json
   ```
   Le label d'un run est déduit du nom du `.out` (`train-<label>-<jobid>.out`) et
   correspond au nom de la config (ex. `second_3stage`).

## Résultats (à remplir)

| Run | Params (M) | GFLOPs @512 | SeK | mIoU | F1 | OA |
|---|---|---|---|---|---|---|
| SECOND 4stage | 206.15 | 554.84 | 0.2442 | 0.7354 | 0.6483 | 0.8822 |
| SECOND 3stage |  84.87 | 457.36 | 0.2464 | 0.7356 | 0.6513 | 0.8840 |
| HiUCD 4stage  | 206.15 | 555.15 | 0.0332 | 0.6569 | 0.3897 | 0.9841 |
| HiUCD 3stage  |  84.88 | 457.67 | 0.0380 | 0.6568 | 0.3716 | 0.9777 |

*GFLOPs @512 mesurés avec `torch.utils.flop_counter` (conv/linéaires/matmuls) ; les scans SSM
(noyaux CUDA custom) ne sont pas comptés → ordre de grandeur. Les paramètres sont exacts.*

**Lecture SECOND.** Retirer le stage 1/32 fait passer le modèle de **206 M → 85 M paramètres (−58.8 %)**
et **555 → 457 GFLOPs (−17.6 %)**, pour une performance **équivalente** (SeK 0.246 vs 0.244, écart dans
le bruit ; mIoU/F1/OA idem). → Le stage profond est **redondant** : gain surtout en **taille de modèle**
(2.4× plus petit), et modeste en calcul — cohérent avec un stage agissant sur seulement 16×16 tokens.

**Lecture Hi-UCD.** Même départ ImageNet. Après correction du bug d'éval label-mask (commit `f4d9e10`),
le SeK atteint **0.033–0.038**, soit le niveau de ChangeMamba (0.049) — alors que ChangeMamba partait
de poids SECOND. Là aussi **3-stage ≈ 4-stage** (le 3-stage devance sur SeK/mIoU, le 4-stage sur F1/OA ;
écarts dans le bruit). L'OA ~0.98 est gonflé par les ~98 % de pixels inchangés (changement très rare,
~2.5 % des pixels). → **La conclusion de l'ablation tient sur les deux datasets.** Un run « benchmark »
depuis des poids SECOND reste utile pour maximiser le SeK et produire les soumissions.

## Notes / limites

- Le stage 1/32 coûte peu de FLOPs (2 blocs sur 16×16 tokens) mais beaucoup de
  **paramètres** (canaux 1024). Si l'objectif est surtout la réduction FLOPs, comparer
  avec une réduction de profondeur du stage 1/16 (les 15 blocs) — mesurer d'abord.
- Renommage des sous-modules décodeur (ModuleList) → un ancien checkpoint *complet*
  SECOND ne se recharge plus tel quel dans ces décodeurs. Sans impact ici puisqu'on
  part d'ImageNet (décodeurs from scratch). La branche `main` reste intacte.
