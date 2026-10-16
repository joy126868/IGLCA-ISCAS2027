# -*- coding: utf-8 -*-
"""How much does the LOLv2-Real score depend on the output scale alpha (gated2: RGB output times alpha)?

The official CIDNet evaluation multiplies the RGB output by alpha = 0.8 (0.8 / 0.82 / 0.84 per checkpoint, chosen on
the test set). For each model this script runs the network once with alpha = 1 on the test images and on every
7th training image, then scores alpha * output (8-bit truncation as in eval) for a grid of alpha values.
Reports PSNR at alpha 1.0 and 0.8, the best alpha on the test set (oracle), and the best alpha on the training
subset together with the test PSNR it gives (a choice made without the test set).
Usage: python analysis/alpha_sweep.py name=weights@arch ...   (arch: a model name as in eval_run.arch_module)
Writes analysis/alpha_sweep.json (or the path in env ALPHA_OUT), with per-image test PSNR for every alpha.
"""
import glob
import importlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
os.chdir(ROOT)
from measure import calculate_psnr  # noqa: E402
from eval_run import arch_module  # noqa: E402

TR = ("LOLV2/LOL-v2/Real_captured/Train/Low", "LOLV2/LOL-v2/Real_captured/Train/Normal")
TE = ("LOLV2/LOL-v2/Real_captured/Test/Low", "LOLV2/LOL-v2/Real_captured/Test/Normal")
ALPHAS = np.round(np.arange(0.70, 1.0001, 0.02), 2)


def sweep(net, dev, low_dir, gt_dir, step=1):
    files = sorted(glob.glob(os.path.join(low_dir, "*.png")))[::step]
    P = np.zeros((len(files), len(ALPHAS)))
    for i, f in enumerate(files):
        x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            y = net(x).clamp(0, 1)[0].permute(1, 2, 0)
        stem = os.path.splitext(os.path.basename(f))[0]
        g = np.asarray(Image.open(os.path.join(gt_dir, stem.replace("low", "normal") + ".png")).convert("RGB"))
        for j, a in enumerate(ALPHAS):
            o = (y * float(a) * 255.0).to(torch.uint8).cpu().numpy()
            P[i, j] = calculate_psnr(o, g)
    return P


def main():
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for spec in sys.argv[1:]:
        name, rest = spec.split("=", 1)
        w, arch = rest.rsplit("@", 1)
        net = importlib.import_module(arch_module(arch)).CIDNet().to(dev).eval()
        net.load_state_dict(torch.load(w, map_location="cpu"))
        net.trans.gated2, net.trans.alpha = False, 1.0
        Pte = sweep(net, dev, *TE)
        Ptr = sweep(net, dev, *TR, step=7)
        te, tr = Pte.mean(0), Ptr.mean(0)
        jt, jr = int(te.argmax()), int(tr.argmax())
        i08, i10 = int(np.argmin(np.abs(ALPHAS - 0.8))), len(ALPHAS) - 1
        out[name] = dict(alphas=ALPHAS.tolist(), test=te.tolist(), train=tr.tolist(), test_per_image=Pte.tolist(),
                         best_alpha_test=float(ALPHAS[jt]), best_alpha_train=float(ALPHAS[jr]))
        print(f"{name}: test PSNR a=1.0 {te[i10]:.2f} | a=0.8 {te[i08]:.2f} | test-best a={ALPHAS[jt]:.2f} {te[jt]:.2f} | "
              f"train-best a={ALPHAS[jr]:.2f} -> test {te[jr]:.2f} (train PSNR a=1.0 {tr[i10]:.2f}, a=0.8 {tr[i08]:.2f})", flush=True)
        del net
        torch.cuda.empty_cache()
    json.dump(out, open(os.environ.get("ALPHA_OUT", os.path.join("analysis", "alpha_sweep.json")), "w"))


if __name__ == "__main__":
    main()
