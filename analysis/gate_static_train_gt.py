# -*- coding: utf-8 -*-
"""Inference-time gate interventions scored in GT-mean PSNR (Fig. 3c; no retraining).

The trained official-loss IG-LCA model is unchanged; only the gate outputs G are replaced at test time:
  learned      : G as learned
  static_train : each channel of G fixed to its mean over the LOLv2-Real TRAINING inputs (no input dependence)
  one          : G = 1 (plain LCA residual with the trained weights)
Scores per test image, as in eval_run.py: PSNR and GT-mean PSNR on 8-bit RGB (gated2 decode, alpha 0.8; GT-mean
rescales the output to the mean gray level of its target, cv2 RGB2GRAY). The training means are saved (--save_means)
so that they can be reused (--load_means). Runs at Idle priority on the E-cores (CPU 16-23).

Usage: python analysis/gate_static_train_gt.py <weights.pth> <out.json> [--save_means f.pt | --load_means f.pt]
"""
import argparse
import ctypes
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from net.CIDNet_IGLCA_ABLATION_3456 import CIDNet  # noqa: E402
from net.IG_LCA import IlluminationGate  # noqa: E402
from measure import calculate_psnr  # noqa: E402

TRAIN_LOW = "LOLV2/LOL-v2/Real_captured/Train/Low"
LOW, GT = "LOLV2/LOL-v2/Real_captured/Test/Low", "LOLV2/LOL-v2/Real_captured/Test/Normal"


def be_nice():
    try:
        k = ctypes.windll.kernel32
        h = k.GetCurrentProcess()
        k.SetPriorityClass(h, 0x40)  # IDLE_PRIORITY_CLASS
        if (os.cpu_count() or 0) >= 24:
            k.SetProcessAffinityMask(h, 0xFF0000)  # E-cores 16-23
    except Exception as e:  # noqa: BLE001
        print("priority/affinity not set:", e)


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save_means", default="")
    ap.add_argument("--load_means", default="")
    a = ap.parse_args()
    be_nice()
    torch.set_num_threads(a.threads)
    model = CIDNet()
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()
    model.trans.gated2 = True
    model.trans.alpha = 0.8
    gates = {n: m for n, m in model.named_modules() if isinstance(m, IlluminationGate)}
    state = {"mode": "collect", "fixed": {}, "collect": {n: [] for n in gates}}

    def hook(mod, inp, out, name):
        if state["mode"] == "collect":
            state["collect"][name].append(out.mean(dim=(0, 2, 3)))
            return None
        if state["mode"] == "fixed":
            return state["fixed"][name].view(1, -1, 1, 1).expand_as(out)
        if state["mode"] == "one":
            return torch.ones_like(out)
        return None

    for n, m in gates.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: hook(mod, inp, out, n))

    t0 = time.time()
    if a.load_means:
        static_train = torch.load(a.load_means)
        ntrain = None
    else:
        files_tr = sorted(glob.glob(os.path.join(TRAIN_LOW, "*.png")))
        state["mode"] = "collect"
        for f in files_tr:
            with torch.no_grad():
                model(load(f))
        static_train = {n: torch.stack(v).mean(0) for n, v in state["collect"].items() if v}
        ntrain = len(files_tr)
        if a.save_means:
            torch.save(static_train, a.save_means)
        print(f"train means from {ntrain} images ({time.time() - t0:.0f} s)", flush=True)

    files = sorted(glob.glob(os.path.join(LOW, "*.png")))
    res = {}
    for mode in ("learned", "static_train", "one"):
        state["mode"] = {"learned": "none", "static_train": "fixed", "one": "one"}[mode]
        state["fixed"] = static_train
        ps, gps = [], []
        for f in files:
            g = np.asarray(Image.open(os.path.join(GT, os.path.basename(f).replace("low", "normal"))).convert("RGB"))
            with torch.no_grad():
                y = model(load(f)).clamp(0, 1)
            o = (y[0].permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
            mo = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).mean()
            mg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY).mean()
            og = np.clip(o.astype(np.float64) * (mg / mo), 0, 255)
            ps.append(float(calculate_psnr(o, g)))
            gps.append(float(calculate_psnr(og, g)))
        res[mode] = dict(psnr=float(np.mean(ps)), gt_psnr=float(np.mean(gps)), per_image_psnr=ps, per_image_gt_psnr=gps)
        print(f"{mode:<13} PSNR {res[mode]['psnr']:.4f}  GT-mean PSNR {res[mode]['gt_psnr']:.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
    res["_meta"] = dict(weights=a.weights, train_images=ntrain, test_images=len(files), seconds=round(time.time() - t0),
                        means_file=a.save_means or a.load_means)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
