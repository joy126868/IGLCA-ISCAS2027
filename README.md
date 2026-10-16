# Brightness or Reconstruction? Revisiting Illumination-Guided Cross-Branch Gating for HVI-Based Low-Light Enhancement

Code and per-image results for the ISCAS 2027 submission.

**Author:** Yi Chiao Pan

The paper studies IG-LCA, an illumination-guided gate on the cross-branch (LCA) residual of
[HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet). The code covers:

- matched multi-seed training of CIDNet, IG-LCA, and the gate variants (official and perceptual-only losses);
- evaluation with PSNR, GT-mean PSNR (brightness-aligned), SSIM, LPIPS, and the brightness error;
- the mechanistic analyses (LayerNorm probe, gate statistics, inference-time gate replacement, hard-image strata);
- the benchmark analyses (output-scale and saturation sweeps, released-weight evaluation);
- the Retinexformer cross-architecture diagnostic (IG-MSA vs. MSA without illumination modulation).

Only the data used by the paper is included: the per-image results of every run in the paper and the
analysis results its tables and figures read, so all numbers, tables, and figures can be regenerated
without retraining (see [Reproducing the paper's numbers](#reproducing-the-papers-numbers)).

## Repository layout

```
net/        CIDNet baseline (CIDNet_LCA_NONE_BASELINE.py), IG-LCA (CIDNet_IGLCA_ABLATION_3456.py),
            gate variants (CIDNet_IGLCA_ABLATION_3456_<variant>.py, IG_LCA_variants.py), HVI transform,
            placement / branch ablation (CIDNet_IGLCA_ABLATION_<blocks>.py, CIDNet_IGLCA_ABLATION_3456_{HV,I}_ONLY.py),
            CIDNet_IGLCA.py (template from which the training scripts generate the CIDNet and IG-LCA modules)
data/       LOL data loaders and default options
loss/       losses (official: L1, SSIM, edge, perceptual on HVI and RGB)
train_ablation_lca_none.py           train CIDNet
train_ablation_iglca3456_FIXED.py    train IG-LCA (gates at LCA3-LCA6; other blocks / one branch with IGLCA_POS / IGLCA_BRANCH)
train_ablation_iglca3456_variant.py  train a gate variant (IGLCA_VARIANT)
train_rf_ablation.py                 Retinexformer diagnostic (ig / noig)
analysis/   evaluation and analysis scripts, analysis results (*.json), per-image results of the final
            checkpoints (per_image/)
results/training/   best-test-epoch logs of the CIDNet runs (used for the checkpoint-selection note)
third_party/        setup and patch scripts for the Retinexformer diagnostic
```

## Environment

Python 3.9, PyTorch 2.7 (CUDA 11.8), torchvision, numpy, opencv-python, scipy, lpips, einops, matplotlib, tqdm,
huggingface_hub (see `requirements.txt`).

```bash
pip install -r requirements.txt
```

## Data

LOLv2-Real and LOLv1, in the layout used by HVI-CIDNet:

```
LOLV2/LOL-v2/Real_captured/Train/{Low,Normal}
LOLV2/LOL-v2/Real_captured/Test/{Low,Normal}
LOL_v1/train/{low,high}
LOL_v1/test/{low,high}
```

## Training

All runs use the official CIDNet settings (Adam, learning rate 1e-4 with cosine decay, 1000 epochs, batch size 8,
256x256 crops) and are scored at the final checkpoint. The seed and the loss are set with environment variables:

| Variable | Values | Meaning |
|---|---|---|
| `IGLCA_SEED` | 3407, 42, 1234 | training seed |
| `IGLCA_RECIPE` | `official`, `percep` | official loss, or the perceptual-only loss |
| `IGLCA_VARIANT` | `iig`, `iigf`, `iige`, `chan`, `scalar`, `se` | gate variant (intensity-only, features + intensity, intensity + global statistics, LayerScale, ReZero, SE) |
| `IGLCA_POS` | `3456` (default); `1`, `2`, `3`, `4`, `5`, `6`, `12`, `34`, `345`, `456`, `1234`, `2345`, `123456` | LCA blocks with the IG-LCA gate (placement ablation, `train_ablation_iglca3456_FIXED.py`) |
| `IGLCA_BRANCH` | empty (default), `hv_only`, `i_only` | gates on one branch only, with `IGLCA_POS=3456` (branch ablation) |

```bash
# CIDNet and IG-LCA, official loss, seed 3407 (LOLv2-Real; use --dataset lol_v1 for LOLv1)
IGLCA_SEED=3407 IGLCA_RECIPE=official python train_ablation_lca_none.py --dataset lolv2_real
IGLCA_SEED=3407 IGLCA_RECIPE=official python train_ablation_iglca3456_FIXED.py --dataset lolv2_real

# a gate variant, e.g. LayerScale
IGLCA_SEED=42 IGLCA_RECIPE=official IGLCA_VARIANT=chan python train_ablation_iglca3456_variant.py --dataset lolv2_real

# architecture ablation (perceptual-only loss, seed 3407): e.g. IG-LCA at LCA6 only, or LCA3-LCA6 on the I branch only
IGLCA_SEED=3407 IGLCA_RECIPE=percep IGLCA_POS=6 python train_ablation_iglca3456_FIXED.py --dataset lolv2_real
IGLCA_SEED=3407 IGLCA_RECIPE=percep IGLCA_BRANCH=i_only python train_ablation_iglca3456_FIXED.py --dataset lolv2_real
```

The run tags are `ablation_iglca_<blocks>_<recipe>_s<seed>` and `ablation_iglca_3456_{hv_only,i_only}_<recipe>_s<seed>`;
`analysis/eval_run.py` maps them to the matching `net/` module. The perceptual-only seed-3407 runs in
`analysis/per_image/` (CIDNet, IG-LCA, the other placements, and the two single-branch models) come from the
exploration that preceded the multi-seed study. They were trained with earlier per-configuration versions of these
training scripts: the same network, loss, and data code, with the DataLoader running in the main process.

On Windows PowerShell, set the variables with `$env:IGLCA_SEED = "3407"` etc. before the command.

## Evaluation

```bash
# per-image PSNR / GT-mean PSNR / SSIM / LPIPS / brightness of a trained run (writes analysis/per_image/<tag>_<ckpt>.json)
python analysis/eval_run.py lolv2_real_ablation_iglca_3456_official_s3407 --ckpt epoch_1000.pth
```

Outputs are decoded as in the official CIDNet test code (output scale alpha = 0.8 on LOLv2-Real); the scripts also
store the scores without the output scale (alpha = 1) and, for LOLv1, without the saturation factor.

## Reproducing the paper's numbers

From the repository root, with the included `analysis/` results:

```bash
python analysis/make_numbers_b.py <out_dir>    # all numbers (numbers_b.tex), Table I (tab_main_b.tex), Table III (tab_gates_b.tex)
                                               # and Table II (tab_ablation_arch.tex)
python analysis/make_fig1_drawio.py <out_dir>/fig1.drawio  # Fig. 1 (export to PDF with draw.io, see the script header)
python analysis/make_fig_b.py <out_dir>/fig_b.pdf      # Fig. 2
python analysis/make_fig_why.py <out_dir>/fig_why.pdf  # Fig. 3
```

The analyses behind these files can be rerun with the scripts in `analysis/` (see the usage line at the top of each
script): `ln_probe.py`, `gate_stats.py`, `gate_intervene.py`, `gate_static_train_gt.py`, `difficulty_strata.py`,
`alpha_sweep.py`, `sat_sweep_lolv1.py`, `eval_released.py`.

## Where each table and figure comes from

All scripts run from the repository root; input paths are relative to `analysis/`. `per_image/<tag>` stands for
`per_image/<tag>_epoch_1000.json` (final checkpoint), `<seed>` for 3407, 42, and 1234.

| Paper | Script | Output | Inputs |
|---|---|---|---|
| Table I, main results | `analysis/make_numbers_b.py <out_dir>` | `tab_main_b.tex` | `per_image/lolv2_real_ablation_{lca_none,iglca_3456}_{percep,official}_s<seed>`; `released_eval/{HVI-CIDNet-LOLv2-real-bestPSNR_a0.8,HVI-CIDNet-LOLv2-real-bestPSNR_a1.0,snr_LOLv2real,retinexformer_LOLv2real}/scores.json` |
| Table II, architecture ablation (gate placement and branch) | `analysis/make_numbers_b.py <out_dir>` | `tab_ablation_arch.tex` | `per_image/lolv2_real_ablation_{lca_none,iglca_<blocks>,iglca_3456_hv_only,iglca_3456_i_only}_percep_s3407`, blocks 1, 2, 3, 4, 5, 6, 12, 34, 345, 456, 1234, 2345, 3456, 123456 |
| Table III, gate designs (PSNR, GT-mean PSNR, SSIM, LPIPS) | `analysis/make_numbers_b.py <out_dir>` | `tab_gates_b.tex` | `per_image/lolv2_real_ablation_iglca_3456_{iig,iigf,iige,chan,scalar,se}_official_s<seed>` (the seeds that exist) and the official-loss CIDNet / IG-LCA runs of Table I |
| Fig. 1, network and IG-LCA block | `analysis/make_fig1_drawio.py <out.drawio>`, then draw.io export to PDF | `.drawio` file | `figures_src/fig1_assets/*.jpg` (an LOL input image, its HVI maps and an enhanced output, taken from our earlier framework figure); connections drawn from the forward pass of `net/CIDNet_IGLCA_ABLATION_3456.py` |
| Fig. 2, brightness analysis | `analysis/make_fig_b.py <out.pdf>` | `fig_b.pdf` | `per_image/lolv2_real_ablation_{lca_none,iglca_3456}_{percep,official}_s<seed>`; `alpha_sweep.json`; `dataset_brightness.json` |
| Fig. 3, mechanistic diagnosis | `analysis/make_fig_why.py <out.pdf>` | `fig_why.pdf` | `ln_probe.json`; `gate_stats_lolv2_real_ablation_iglca_3456_official_s<seed>.json`; `gate_static_train_gt_lolv2_real_ablation_iglca_3456_official_s<seed>.json`; `per_image/lolv2_real_ablation_{lca_none,iglca_3456,iglca_3456_iig,iglca_3456_iigf,iglca_3456_iige}_official_s<seed>` |
| Numbers in the text | `analysis/make_numbers_b.py <out_dir>` | `numbers_b.tex` | the inputs above, plus `gate_intervene_lolv2_real_ablation_iglca_3456_official_s<seed>.json`, `gate_stats_s3407_e1000.json`, `gate_stats_lolv2_real_ablation_iglca_3456_percep_s{42,1234}.json`, `difficulty_strata_{official,percep}.json`, `test_difficulty.json`, `sat_sweep_lolv1.json`, `dataset_*.json`, the LOLv1 and Retinexformer per-image files, `per_image/lolv2_real_ablation_iglca_3456_scalar_percep_s42`, and `results/training/*/best_metrics_*.log` |

## Model selection and forward-pass notes

- LCA3-LCA6 was selected from the single-seed, perceptual-only placement ablation (Table II) by test-set PSNR, so
  this choice is exploratory. The multi-seed comparisons of Tables I and III use this placement fixed.
- The official CIDNet forward pass, kept unchanged here, discards the output of the I-branch LCA5: the next line
  overwrites it, so that gate receives no gradient. With gates at LCA3-LCA6, 7 of the 8 gates are therefore effective.
  The LCA2 outputs reach only the skip connection, because the next encoder stage reads the block-2 output rather than
  the LCA2 output.

## Retinexformer diagnostic

1. Clone the official [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) repository.
2. `python third_party/setup_rf_copy.py <path to the clone>` copies the code to `third_party/Retinexformer_iscas` and
   builds `datasets_rf/LOLv2_real` with hard links to the LOLv2-Real images.
3. `python third_party/patch_rf_ig.py` adds the switch that removes the illumination modulation of IG-MSA
   (light-up retained).
4. Train with `IGLCA_SEED=<seed> IGLCA_VARIANT=ig128` or `noig128` and `python train_rf_ablation.py`, then evaluate
   with `python analysis/eval_rf.py <tag>`.

## Model weights

The trained weights are not included in this repository because of their size.

## License

MIT (see `LICENSE`). The code builds on HVI-CIDNet and Retinexformer, both released under the MIT License; their
license notices are reproduced in `THIRD_PARTY_LICENSES.md`.
