# -*- coding: utf-8 -*-
"""Is the LOLv1 decoding constant of CIDNet test-tuned, like alpha on LOLv2-Real? (calibration audit)

The official CIDNet test code decodes LOLv1 outputs with trans.gated = True, which multiplies the HVI saturation by
alpha_s = 1.3 before the inverse transform (net/HVI_transform.py, PHVIT). LOLv1 has no validation split either.
For each model the network runs once per image (the HVI output is captured at the PHVIT call); the output is then
decoded for a grid of saturation scales s and scored on the 15 test images and on every 5th training image.
Reports PSNR / GT-mean PSNR at s = 1 (no gate) and s = 1.3 (official), and the best s on the test and on the
training images.
Usage: python analysis/sat_sweep_lolv1.py name=weights[@arch] ...   (.safetensors or .pth; arch: eval_run model name)
Writes analysis/sat_sweep_lolv1.json.
"""
import glob
import importlib
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
os.chdir(ROOT)
from measure import calculate_psnr  # noqa: E402
from eval_run import arch_module  # noqa: E402

S_GRID = [round(1.0 + 0.05 * i, 2) for i in range(13)]  # 1.00 ... 1.60
SPLITS = {"test": ("LOL_v1/test/low", "LOL_v1/test/normal", 1), "train": ("LOL_v1/train/low", "LOL_v1/train/high", 5)}


def load(spec, dev):
    w, arch = (spec.split("@") + ["lca_none"])[:2]
    net = importlib.import_module(arch_module(arch)).CIDNet().to(dev).eval()
    if w.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(w)
    else:
        sd = torch.load(w, map_location="cpu")
    net.load_state_dict(sd)
    net.trans.gated, net.trans.gated2 = False, False
    return net


def gt_mean_psnr(o, g):
    mo = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).mean()
    mg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY).mean()
    return calculate_psnr(np.clip(o.astype(np.float64) * (mg / max(mo, 1e-6)), 0, 255), g)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for arg in sys.argv[1:]:
        name, spec = arg.split("=", 1)
        net = load(spec, dev)
        captured = {}
        phvit = net.trans.PHVIT

        def capture(hvi):
            captured["hvi"] = hvi
            return phvit(hvi)
        net.trans.PHVIT = capture
        out = {"weights": spec, "s_grid": S_GRID}
        for split, (low, gtd, step) in SPLITS.items():
            files = sorted(glob.glob(os.path.join(low, "*.png")))[::step]
            P = np.zeros((len(files), len(S_GRID)))
            PG = np.zeros_like(P)
            for i, f in enumerate(files):
                x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
                g = np.asarray(Image.open(os.path.join(gtd, os.path.basename(f))).convert("RGB"))
                with torch.no_grad():
                    net(x)
                    hvi = captured["hvi"]
                    for j, s in enumerate(S_GRID):
                        net.trans.gated, net.trans.alpha_s = s != 1.0, s
                        y = phvit(hvi).clamp(0, 1)
                        o = (y[0].permute(1, 2, 0) * 255.0).to(torch.uint8).cpu().numpy()
                        P[i, j], PG[i, j] = calculate_psnr(o, g), gt_mean_psnr(o, g)
                    net.trans.gated, net.trans.alpha_s = False, 1.3
            mp, mpg = P.mean(0), PG.mean(0)
            j13 = S_GRID.index(1.3)
            out[split] = dict(n=len(files), psnr=mp.tolist(), gt_psnr=mpg.tolist(),
                              best_s_psnr=S_GRID[int(mp.argmax())], best_s_gt=S_GRID[int(mpg.argmax())],
                              psnr_s1=float(mp[0]), psnr_s13=float(mp[j13]), gt_s1=float(mpg[0]), gt_s13=float(mpg[j13]),
                              per_image_psnr=P.tolist() if split == "test" else None)
            print(f"{name:28s} {split:5s} n={len(files):3d}  PSNR s=1 {mp[0]:.2f}  s=1.3 {mp[j13]:.2f}  best s {out[split]['best_s_psnr']}"
                  f" ({mp.max():.2f}) | GT-mean s=1 {mpg[0]:.2f}  s=1.3 {mpg[j13]:.2f}  best s {out[split]['best_s_gt']}")
        res[name] = out
    json.dump(res, open(os.path.join("analysis", "sat_sweep_lolv1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
