# -*- coding: utf-8 -*-
"""Representation probe: how much absolute-intensity information survives the LayerNorm in front of the gate?

Every IG-LCA gate reads LN(F_I), F_I = the I-branch feature at that level (net/IG_LCA.py: the HV update uses norm(y),
the I update norm(x); both are the input of I_LCA<k>). For the official-loss models (CIDNet baseline and IG-LCA-3456,
seeds 3407 / 42 / 1234) and the gated levels k = 3..6, three per-pixel feature sets predict the log input intensity
log L_down (L = max(R, G, B) of the input, average-pooled to the level's resolution):
  pre     : F_I                                   (C numbers)
  post    : the block's LN(F_I), i.e. what the gate reads (C numbers; LN's affine part is linear)
  removed : channel mean and std of F_I per pixel (the two numbers LN takes away)
Design (no test-set fitting, no pixel-level random split):
  fit  : every 5th LOLv2-Real TRAINING input in sorted order (138 images)
  test : all 100 LOLv2-Real test inputs (never used for fitting)
  ridge regression, lambda = 1 on features standardized with fit-set statistics; PIXELS sampled per image (seeded).
Reported on the test images: R2 overall, and within-image R2 (target and prediction centred per image: does the feature
say WHERE in the image it is bright, beyond the scene exposure?).
This is representation evidence (linear decodability), not a causal test; whether the gate uses the information is
judged together with the gate statistics and the gate-replacement tests (analysis/gate_stats.py, gate_intervene.py).
Usage: python analysis/ln_probe.py [--pixels 1500] [--threads 4] [--limit N]     Writes analysis/ln_probe.json.
"""
import argparse
import glob
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
os.chdir(ROOT)
from eval_run import arch_module  # noqa: E402

TRAIN_LOW = "LOLV2/LOL-v2/Real_captured/Train/Low"
TEST_LOW = "LOLV2/LOL-v2/Real_captured/Test/Low"
FIT_STEP = 5
LAMBDA = 1.0
MODELS = [(m, s) for m in ("lca_none", "iglca_3456") for s in ("3407", "42", "1234")]
LEVELS = (3, 4, 5, 6)
SETS = ("pre", "post", "removed")


def load_net(model, seed, dev):
    net = importlib.import_module(arch_module(model)).CIDNet().to(dev).eval()
    w = os.path.join("weights", f"lolv2_real_ablation_{model}_official_s{seed}", "epoch_1000.pth")
    net.load_state_dict(torch.load(w, map_location="cpu"))
    return net


def features(net, files, n_pix, seed):
    """Per level: sampled per-pixel feature sets, targets and image ids for the given input files."""
    cap = {}
    hooks = [getattr(net, f"I_LCA{k}").register_forward_pre_hook(
        (lambda k: (lambda mod, args: cap.__setitem__(k, args[0].detach())))(k)) for k in LEVELS]
    data = {k: {n: [] for n in SETS + ("y", "img")} for k in LEVELS}
    rng = np.random.default_rng(seed)
    for ii, f in enumerate(files):
        x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            net(x)
            L = x.max(dim=1, keepdim=True)[0]
            for k in LEVELS:
                fi = cap[k]
                post = getattr(net, f"I_LCA{k}").norm(fi)
                h, w = fi.shape[-2:]
                y = torch.log(F.adaptive_avg_pool2d(L, (h, w)) + 1e-3).flatten()
                pre_v, post_v = fi[0].flatten(1).t(), post[0].flatten(1).t()
                rem = torch.stack([pre_v.mean(1), pre_v.std(1, unbiased=False)], 1)
                idx = rng.choice(h * w, size=min(n_pix, h * w), replace=False)
                d = data[k]
                for name, v in (("pre", pre_v), ("post", post_v), ("removed", rem), ("y", y)):
                    d[name].append(v[idx].numpy().astype(np.float64))
                d["img"].append(np.full(len(idx), ii))
    for h_ in hooks:
        h_.remove()
    return {k: {n: np.concatenate(v) for n, v in d.items()} for k, d in data.items()}


def fit_eval(Xf, yf, Xt, yt, img_t):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    A, B = (Xf - mu) / sd, (Xt - mu) / sd
    ym = yf.mean()
    w = np.linalg.solve(A.T @ A + LAMBDA * np.eye(A.shape[1]), A.T @ (yf - ym))
    pred = B @ w + ym
    r2 = 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)
    yc, pc = yt.copy(), pred.copy()
    for i in np.unique(img_t):
        m = img_t == i
        yc[m] -= yt[m].mean()
        pc[m] -= pred[m].mean()
    return float(r2), float(1 - np.sum((yc - pc) ** 2) / np.sum(yc ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: first N fit and test images, one model per kind")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fit_files = sorted(glob.glob(os.path.join(TRAIN_LOW, "*.png")))[::FIT_STEP]
    test_files = sorted(glob.glob(os.path.join(TEST_LOW, "*.png")))
    models = MODELS
    if a.limit:
        fit_files, test_files, models = fit_files[:a.limit], test_files[:a.limit], [("lca_none", "3407"), ("iglca_3456", "3407")]
    meta = dict(fit=f"every {FIT_STEP}th LOLv2-Real training input ({len(fit_files)} images)",
                test=f"all LOLv2-Real test inputs ({len(test_files)} images)", pixels_per_image=a.pixels, ridge_lambda=LAMBDA)
    out = {"meta": meta}
    print(meta, flush=True)
    for model, seed in models:
        net = load_net(model, seed, "cpu")  # forward only; CPU leaves the GPU to the trainings
        Df, Dt = features(net, fit_files, a.pixels, 1), features(net, test_files, a.pixels, 2)
        res = {}
        for k in LEVELS:
            r = {n: fit_eval(Df[k][n], Df[k]["y"], Dt[k][n], Dt[k]["y"], Dt[k]["img"]) for n in SETS}
            res[f"LCA{k}"] = {n: {"r2": v[0], "r2_within": v[1]} for n, v in r.items()}
            print(f"{model:10s} s{seed:4s} LCA{k} C={Df[k]['pre'].shape[1]:3d} | test R2 pre {r['pre'][0]:.3f} "
                  f"post {r['post'][0]:.3f} removed {r['removed'][0]:.3f} | within-image pre {r['pre'][1]:.3f} "
                  f"post {r['post'][1]:.3f} removed {r['removed'][1]:.3f}", flush=True)
        out[f"{model}_s{seed}"] = res
        json.dump(out, open(os.path.join("analysis", "ln_probe.json" if not a.limit else "ln_probe_smoke.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
