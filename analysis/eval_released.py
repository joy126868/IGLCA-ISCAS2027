# -*- coding: utf-8 -*-
"""Score a CIDNet checkpoint (released safetensors or our .pth) on a LOL test set with the training-time
evaluation pipeline: official decode (LOLv2-Real: gated2 + alpha; LOLv1: gated), 8-bit outputs through
torchvision ToPILImage, measure.metrics (PSNR/SSIM on 8-bit RGB, AlexNet LPIPS), with and without GT-mean.

Usage: python analysis/eval_released.py <weights> <dataset lolv2_real|lol_v1> <alpha> <out_dir> [--arch base|iglca]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from measure import metrics  # noqa: E402

SETS = {"lolv2_real": ("LOLV2/LOL-v2/Real_captured/Test/Low", "LOLV2/LOL-v2/Real_captured/Test/Normal/"),
        "lol_v1": ("LOL_v1/test/low", "LOL_v1/test/normal/")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("dataset", choices=sorted(SETS))
    ap.add_argument("alpha", type=float)
    ap.add_argument("out_dir")
    ap.add_argument("--arch", default="base", choices=["base", "iglca"])
    a = ap.parse_args()
    if a.arch == "base":
        from net.CIDNet_LCA_NONE_BASELINE import CIDNet
    else:
        from net.CIDNet_IGLCA_ABLATION_3456 import CIDNet
    model = CIDNet().cuda().eval()
    if a.weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(a.weights)
    else:
        state = torch.load(a.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True), None
    if a.dataset == "lolv2_real":
        model.trans.gated2 = True
        model.trans.alpha = a.alpha
    else:
        model.trans.gated = True
    low, gt = SETS[a.dataset]
    os.makedirs(a.out_dir, exist_ok=True)
    to_t = transforms.ToTensor()
    for f in sorted(glob.glob(os.path.join(low, "*.png"))):
        x = to_t(Image.open(f).convert("RGB"))[None].cuda()
        with torch.no_grad():
            y = model(x).clamp(0, 1)
        transforms.ToPILImage()(y[0].cpu()).save(os.path.join(a.out_dir, os.path.basename(f)))
    res = {}
    for gm in (False, True):
        p, s, l = metrics(os.path.join(a.out_dir, "*.png"), gt, use_GT_mean=gm)
        res["gtmean" if gm else "plain"] = dict(psnr=float(p), ssim=float(s), lpips=float(l))
    res["_meta"] = dict(weights=a.weights, dataset=a.dataset, alpha=a.alpha)
    json.dump(res, open(os.path.join(a.out_dir, "scores.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
