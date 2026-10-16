# -*- coding: utf-8 -*-
"""Retinexformer illumination-guidance ablation (cross-architecture diagnostic of the paper).

Question: does the illumination guidance of a PUBLISHED design (Retinexformer's IG-MSA, v = v * illu_fea) improve
structure, or, like IG-LCA, mostly global brightness? ig = official network; noig = the same network without that
multiplication (third_party/Retinexformer_iscas, patched by third_party/patch_rf_ig.py; no parameter changes).

Launch like the CIDNet runs (queue line):  train_rf_ablation.py <seed> lolv2_real <ig|noig>[crop]
(the paper uses ig128 / noig128 = 128-pixel crops, as in the official LOLv1 recipe)
run_seed.ps1 passes --dataset/--threads and sets IGLCA_SEED, IGLCA_VARIANT and IGLCA_RECIPE (official only).
The option file is the copy's official Options/RetinexFormer_LOL_v2_real.yml with only these changes:
  name, manual_seed, network_g.ig, data roots (hard links to this project's LOLv2-Real, same images as the CIDNet runs),
  num_worker_per_gpu = --threads, no tensorboard, validation and checkpoints every 5k instead of 1k iterations
  (logging only: the recipe -- network, 256 crops, batch 8, mixup, L1, Adam 2e-4, cyclic cosine, 150k iterations --
  is unchanged). Training runs the copy's unmodified basicsr/train.py.
Outputs: third_party/Retinexformer_iscas/experiments/RF_LOLv2real_<variant>_s<seed>/models/net_g_150000.pth
Debug: --total-iter 4 (batch 2, 64 px crops, no validation) for a pipeline smoke test.
"""
import argparse
import os
import runpy
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "Retinexformer_iscas")
STUBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "rf_stubs")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_rf", "LOLv2_real")


def option_text(variant, seed, threads, total_iter):
    """variant = ig|noig plus an optional crop size, e.g. noig128 (default 256 = official LOLv2-Real crop)."""
    base = variant.rstrip("0123456789")
    crop = int(variant[len(base):] or 256)
    name = f"RF_LOLv2real_{variant}_s{seed}" + ("_debug" if total_iter else "")
    s = open(os.path.join(ROOT, "Options", "RetinexFormer_LOL_v2_real.yml"), encoding="utf-8").read()
    repl = [
        ("name: Enhancement_RetinexFormer_LOL_v2_real", f"name: {name}"),
        ("manual_seed: 100", f"manual_seed: {seed}"),
        ("dataroot_gt: data/LOLv2/Real_captured/Train/Normal", f"dataroot_gt: {DATA}/Train/target"),
        ("dataroot_lq: data/LOLv2/Real_captured/Train/Low", f"dataroot_lq: {DATA}/Train/input"),
        ("dataroot_gt: data/LOLv2/Real_captured/Test/Normal", f"dataroot_gt: {DATA}/Test/target"),
        ("dataroot_lq: data/LOLv2/Real_captured/Test/Low", f"dataroot_lq: {DATA}/Test/input"),
        ("num_worker_per_gpu: 8", f"num_worker_per_gpu: {threads}"),
        ("use_tb_logger: true", "use_tb_logger: false"),
        ("val_freq: !!float 1e3", "val_freq: !!float 5e3"),
        ("save_checkpoint_freq: !!float 1e3", "save_checkpoint_freq: !!float 5e3"),
        ("  num_blocks: [1,2,2]", f"  num_blocks: [1,2,2]\n  ig: {'true' if base == 'ig' else 'false'}"),
    ]
    if crop != 256:
        repl += [("gt_size: 256", f"gt_size: {crop}"), ("gt_sizes: [256]", f"gt_sizes: [{crop}]")]
    if total_iter:  # pipeline smoke test only
        repl += [("total_iter: 150000", f"total_iter: {total_iter}"),
                 ("batch_size_per_gpu: 8", "batch_size_per_gpu: 2"), ("mini_batch_sizes: [8]", "mini_batch_sizes: [2]"),
                 (f"gt_size: {crop}", "gt_size: 64"), (f"gt_sizes: [{crop}]", "gt_sizes: [64]"),
                 ("val_freq: !!float 5e3", "val_freq: !!float 1e9"),
                 ("save_checkpoint_freq: !!float 5e3", f"save_checkpoint_freq: {total_iter}"),
                 ("print_freq: 500", "print_freq: 1")]
    for old, new in repl:
        n = s.count(old)
        if n != 1:
            raise RuntimeError(f"option pattern found {n} times: {old!r}")
        s = s.replace(old, new)
    return name, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lolv2_real")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--total-iter", type=int, default=0, help="pipeline smoke test only")
    a, _ = ap.parse_known_args()
    seed = int(os.environ.get("IGLCA_SEED", "3407"))
    variant = os.environ.get("IGLCA_VARIANT", "ig")
    recipe = os.environ.get("IGLCA_RECIPE", "official")
    if a.dataset != "lolv2_real" or variant.rstrip("0123456789") not in ("ig", "noig") or recipe != "official":
        raise SystemExit(f"unsupported run: dataset={a.dataset} variant={variant} recipe={recipe}")
    name, text = option_text(variant, seed, a.threads, a.total_iter)
    opt_dir = os.path.join(ROOT, "Options", "iscas")
    os.makedirs(opt_dir, exist_ok=True)
    opt_path = os.path.join(opt_dir, name + ".yml")
    open(opt_path, "w", encoding="utf-8").write(text)
    print(f"===> {name}: option file {opt_path}", flush=True)
    # the official basicsr/train.py of the copy. The copy's basicsr has an empty __init__.py (setup_rf_copy.py), so with
    # ROOT first on sys.path it wins over the unrelated pip basicsr of the env.
    sys.path.insert(0, STUBS)
    sys.path.insert(0, ROOT)
    # DataLoader workers are spawned on Windows and do not see these sys.path changes; PYTHONPATH reaches them
    os.environ["PYTHONPATH"] = os.pathsep.join([ROOT, STUBS] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    os.chdir(ROOT)
    import basicsr
    if not os.path.abspath(basicsr.__file__).startswith(ROOT):
        raise RuntimeError(f"wrong basicsr imported: {basicsr.__file__}")
    sys.argv = [os.path.join(ROOT, "basicsr", "train.py"), "--opt", opt_path]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
