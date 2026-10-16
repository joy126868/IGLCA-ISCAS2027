# -*- coding: utf-8 -*-
"""Test-time interventions on the learned IG-LCA gates (mechanism analysis, reviewer #27).

The trained model is kept fixed; only the gate outputs G are replaced:
  learned  : G as learned (should reproduce the training-log score of the checkpoint)
  one      : G = 1 everywhere (plain LCA residual with the trained weights)
  chmean   : per image, each channel's G replaced by its spatial mean (no spatial variation)
  static   : each channel's G replaced by its mean over the whole test set (no input dependence)
  spmean   : per image, G replaced by its channel mean (spatial map only, no channel variation)
Scores: PSNR / SSIM on 8-bit RGB (measure.py, as in training), LOLv2-real test, gated2 decode, alpha 0.8.

Usage: python analysis/gate_intervene.py <weights.pth> <out.json> [--threads 4] [--modes learned,one,...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from net.CIDNet_IGLCA_ABLATION_3456 import CIDNet  # noqa: E402
from net.IG_LCA import IlluminationGate  # noqa: E402
from measure import calculate_psnr, calculate_ssim  # noqa: E402

LOW, GT = "LOLV2/LOL-v2/Real_captured/Test/Low", "LOLV2/LOL-v2/Real_captured/Test/Normal"


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--modes", default="learned,one,chmean,static,spmean")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = CIDNet()
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()
    model.trans.gated2 = True
    model.trans.alpha = 0.8
    gates = {n: m for n, m in model.named_modules() if isinstance(m, IlluminationGate)}
    state = {"mode": "learned", "static": {}, "collect": {n: [] for n in gates}}

    def hook(mod, inp, out, name):
        mode = state["mode"]
        if mode == "learned":
            state["collect"][name].append(out.mean(dim=(0, 2, 3)))  # per-channel spatial mean
            return None
        if mode == "one":
            return torch.ones_like(out)
        if mode == "chmean":
            return out.mean(dim=(2, 3), keepdim=True).expand_as(out)
        if mode == "static":
            return state["static"][name].view(1, -1, 1, 1).expand_as(out)
        if mode == "spmean":
            return out.mean(dim=1, keepdim=True).expand_as(out)
        raise KeyError(mode)

    for n, m in gates.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: hook(mod, inp, out, n))

    files = sorted(glob.glob(os.path.join(LOW, "*.png")))
    if a.limit:
        files = files[:a.limit]
    modes = a.modes.split(",")
    if "static" in modes and "learned" not in modes:
        modes = ["learned"] + modes
    if "static" in modes:  # static needs the learned per-channel means first
        modes = ["learned"] + [m for m in modes if m != "learned"]
    res = {}
    for mode in modes:
        state["mode"] = mode
        ps, ss = [], []
        for f in files:
            gt = np.asarray(Image.open(os.path.join(GT, os.path.basename(f).replace("low", "normal"))).convert("RGB"))
            with torch.no_grad():
                y = model(load(f)).clamp(0, 1)
            # same 8-bit conversion as training eval (torchvision ToPILImage: mul(255).byte(), truncation)
            o = (y[0].permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
            ps.append(calculate_psnr(o, gt))
            ss.append(calculate_ssim(o, gt))
        res[mode] = dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), per_image_psnr=[float(p) for p in ps])
        print(f"{mode:<8} PSNR {res[mode]['psnr']:.4f}  SSIM {res[mode]['ssim']:.4f}", flush=True)
        if mode == "learned":
            state["static"] = {n: torch.stack(v).mean(0) for n, v in state["collect"].items() if v}
    res["_meta"] = dict(weights=a.weights, images=len(files))
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
