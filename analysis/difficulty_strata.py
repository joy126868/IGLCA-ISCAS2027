# -*- coding: utf-8 -*-
"""Difficulty-stratified comparison of CIDNet and IG-LCA-3456 on LOLv2-Real (answers: dataset difficulty, model weakness,
or brightness?).

Per test image: input mean gray level (exposure), relative input noise (Immerkaer's fast noise estimate on the gray input
divided by its mean), and scene detail (mean gradient magnitude of the gray target). For each factor the images are split
into terciles. Per tercile, pooled over the seeds where both models exist:
  - the absolute scores of CIDNet (PSNR, GT-mean PSNR): do both models get worse on hard images?
  - IG-LCA minus CIDNet in PSNR and in GT-mean PSNR, with a bootstrap interval that resamples seeds and then images:
    does the gate add a weakness of its own, and is it structural (GT-mean) or brightness (PSNR only)?
Usage: python analysis/difficulty_strata.py [--recipe official] [--seeds 3407 42 1234]
Writes analysis/test_difficulty.json (per-image factors) and analysis/difficulty_strata.json (tercile summaries).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image

LOW, GT = "LOLV2/LOL-v2/Real_captured/Test/Low", "LOLV2/LOL-v2/Real_captured/Test/Normal"
PI = os.path.join("analysis", "per_image")
KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def immerkaer_sigma(gray):
    """Immerkaer (1996): sigma = sqrt(pi/2) / (6 (W-2)(H-2)) * sum |gray * N|."""
    h, w = gray.shape
    r = cv2.filter2D(gray, -1, KERNEL, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(r).sum() / (6.0 * (w - 2) * (h - 2)))


def factors():
    f = os.path.join("analysis", "test_difficulty.json")
    if os.path.exists(f):
        return json.load(open(f))
    out = {}
    for p in sorted(glob.glob(os.path.join(LOW, "*.png"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        x = cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float64)
        g = cv2.cvtColor(np.asarray(Image.open(os.path.join(GT, stem.replace("low", "normal") + ".png")).convert("RGB")),
                         cv2.COLOR_RGB2GRAY).astype(np.float64)
        gx, gy = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        out[stem] = dict(input_mean=float(x.mean()), rel_noise=immerkaer_sigma(x) / max(x.mean(), 1e-6),
                         detail=float(np.sqrt(gx ** 2 + gy ** 2).mean()))
    json.dump(out, open(f, "w"), indent=1)
    return out


def load(model, recipe, seed):
    f = os.path.join(PI, f"lolv2_real_ablation_{model}_{recipe}_s{seed}_epoch_1000.json")
    return json.load(open(f))["images"] if os.path.exists(f) else None


def hboot(diffs_by_seed, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    D = [np.asarray(d) for d in diffs_by_seed]
    M = np.stack([d[rng.integers(0, len(d), (B, len(d)))].mean(1) for d in D], axis=1)
    pick = rng.integers(0, len(D), (B, len(D)))
    boot = np.take_along_axis(M, pick, axis=1).mean(1)
    return [float(v) for v in np.percentile(boot, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="official")
    ap.add_argument("--seeds", nargs="+", default=["3407", "42", "1234"])
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    F = factors()
    seeds = [s for s in a.seeds if load("lca_none", a.recipe, s) and load("iglca_3456", a.recipe, s)]
    A = {s: load("lca_none", a.recipe, s) for s in seeds}
    B = {s: load("iglca_3456", a.recipe, s) for s in seeds}
    stems = sorted(F)
    res = {}
    labels = {"input_mean": ("dark input", "", "bright input"), "rel_noise": ("low noise", "", "high noise"),
              "detail": ("little detail", "", "much detail")}
    print(f"recipe {a.recipe}, seeds {seeds}")
    for fac in ("input_mean", "rel_noise", "detail"):
        v = np.array([F[k][fac] for k in stems])
        q = np.quantile(v, [1 / 3, 2 / 3])
        groups = [[k for k in stems if F[k][fac] <= q[0]], [k for k in stems if q[0] < F[k][fac] <= q[1]],
                  [k for k in stems if F[k][fac] > q[1]]]
        res[fac] = []
        print(f"== {fac} (tercile edges {q[0]:.3g}, {q[1]:.3g})")
        for gi, ks in enumerate(groups):
            cp = float(np.mean([A[s][k]["psnr"] for s in seeds for k in ks]))
            cg = float(np.mean([A[s][k]["gt_psnr"] for s in seeds for k in ks]))
            ip = float(np.mean([B[s][k]["psnr"] for s in seeds for k in ks]))
            ig = float(np.mean([B[s][k]["gt_psnr"] for s in seeds for k in ks]))
            dps = [[B[s][k]["psnr"] - A[s][k]["psnr"] for k in ks] for s in seeds]
            dgt = [[B[s][k]["gt_psnr"] - A[s][k]["gt_psnr"] for k in ks] for s in seeds]
            cbe = float(np.mean([abs(np.log(A[s][k]["bright"])) for s in seeds for k in ks]))
            row = dict(group=gi, n=len(ks), cid_psnr=cp, cid_gt=cg, ig_psnr=ip, ig_gt=ig, cid_berr=cbe,
                       d_psnr=float(np.mean(dps)), d_psnr_ci=hboot(dps), d_gt=float(np.mean(dgt)), d_gt_ci=hboot(dgt))
            res[fac].append(row)
            lab = labels[fac][gi] or "middle"
            print(f"  {lab:14s} n={len(ks):3d}  CIDNet PSNR {cp:5.2f} GT {cg:5.2f} berr {cbe:.3f} | IG PSNR {ip:5.2f} GT {ig:5.2f} | "
                  f"dPSNR {row['d_psnr']:+.2f} [{row['d_psnr_ci'][0]:+.2f},{row['d_psnr_ci'][1]:+.2f}]  "
                  f"dGT {row['d_gt']:+.2f} [{row['d_gt_ci'][0]:+.2f},{row['d_gt_ci'][1]:+.2f}]")
    json.dump(dict(recipe=a.recipe, seeds=seeds, strata=res), open(os.path.join("analysis", f"difficulty_strata_{a.recipe}.json"), "w"), indent=1)
    # how strongly the factors are related to each other
    v = np.array([[F[k]["input_mean"], F[k]["rel_noise"], F[k]["detail"]] for k in stems])
    from scipy import stats
    print("Spearman input_mean~rel_noise %.2f, input_mean~detail %.2f, rel_noise~detail %.2f" % (
        stats.spearmanr(v[:, 0], v[:, 1])[0], stats.spearmanr(v[:, 0], v[:, 2])[0], stats.spearmanr(v[:, 1], v[:, 2])[0]))


if __name__ == "__main__":
    main()
