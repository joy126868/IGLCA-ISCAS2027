# -*- coding: utf-8 -*-
"""Per-image scores of one trained run (for paired tests between models of the same seed).

Usage: python analysis/eval_run.py <run_tag> [--ckpt epoch_1000.pth|best] [--weights path --arch base|iglca|<variant>]
Run tag format: <dataset>_ablation_<model>[_<variant>]_<recipe>_s<seed>, weights in weights/<run_tag>/.
Decode as in training evaluation: LOLv2-Real gated2 + alpha 0.8, LOLv1 gated. Outputs are converted to
8 bit like torchvision's ToPILImage (truncation). Writes analysis/per_image/<run_tag>_<ckpt>.json with
per-image PSNR, SSIM, GT-mean PSNR/SSIM (measure.py functions) the output/GT mean-gray ratio (bright)
and LPIPS (AlexNet, as measure.py);
LOLv2-Real also gets psnr_a1, the PSNR of the same output without the alpha = 0.8 scaling.
"""
import argparse
import glob
import importlib
import json
import os
import re
import sys

import cv2
import lpips
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from measure import calculate_psnr, calculate_ssim  # noqa: E402

SETS = {"lolv2_real": ("LOLV2/LOL-v2/Real_captured/Test/Low", "LOLV2/LOL-v2/Real_captured/Test/Normal", lambda s: s.replace("low", "normal")),
        "lol_v1": ("LOL_v1/test/low", "LOL_v1/test/normal", lambda s: s)}


def arch_module(model):
    model = re.sub(r"_ft$", "", model)  # fine-tuned runs use the architecture of their model
    if model == "lca_none":
        return "net.CIDNet_LCA_NONE_BASELINE"
    if model == "iglca_3456":
        return "net.CIDNet_IGLCA_ABLATION_3456"
    m = re.match(r"iglca_(\d+)$", model)
    if m:
        return f"net.CIDNet_IGLCA_ABLATION_{m.group(1)}"
    if model in ("iglca_3456_hv_only", "iglca_3456_i_only"):
        return "net.CIDNet_IGLCA_ABLATION_3456_" + model[len("iglca_3456_"):].upper()
    m = re.match(r"iglca_(\d+)_(\w+)$", model)  # variant script: iglca_<positions>_<variant>
    if m:
        return f"net.CIDNet_IGLCA_ABLATION_{m.group(1)}_{m.group(2)}"
    raise KeyError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ckpt", default="epoch_1000.pth")
    ap.add_argument("--weights", default="")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    m = re.match(r"(lolv2_real|lol_v1)_ablation_(.+)_(official|percep)_s(\d+)$", a.tag)
    ds, model = m.group(1), m.group(2)
    wpath = a.weights or (os.path.join("weights", a.tag, a.ckpt) if a.ckpt != "best"
                          else glob.glob(os.path.join("weights", a.tag, "PSNR_*_bestweight.pth"))[0])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = importlib.import_module(arch_module(model)).CIDNet().to(dev).eval()
    net.load_state_dict(torch.load(wpath, map_location="cpu"))
    if ds == "lolv2_real":
        net.trans.gated2, net.trans.alpha = True, 0.8
    else:
        net.trans.gated = True
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()  # as measure.metrics (AlexNet, lpips.im2tensor)
    low, gtd, gname = SETS[ds]
    rows = {}
    for f in sorted(glob.glob(os.path.join(low, "*.png"))):
        x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            y = net(x).clamp(0, 1)
        o = (y[0].permute(1, 2, 0) * 255.0).to(torch.uint8).cpu().numpy()
        stem = os.path.splitext(os.path.basename(f))[0]
        g = np.asarray(Image.open(os.path.join(gtd, gname(stem) + ".png")).convert("RGB"))
        mo = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).mean()
        mg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY).mean()
        og = np.clip(o.astype(np.float64) * (mg / mo), 0, 255)
        with torch.no_grad():
            lpv = float(lp(lpips.im2tensor(g).to(dev), lpips.im2tensor(o).to(dev)).item())
        extra = {}
        if ds == "lolv2_real":  # the same output without the alpha = 0.8 scaling (alpha * rgb is exact, rgb <= 1)
            o1 = (y[0].permute(1, 2, 0) / 0.8 * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            extra = dict(psnr_a1=float(calculate_psnr(o1, g)))
        else:
            net.trans.gated = False
            with torch.no_grad():
                y1 = net(x).clamp(0, 1)
            net.trans.gated = True
            o1 = (y1[0].permute(1, 2, 0) * 255.0).to(torch.uint8).cpu().numpy()
            m1 = cv2.cvtColor(o1, cv2.COLOR_RGB2GRAY).mean()
            og1 = np.clip(o1.astype(np.float64) * (mg / m1), 0, 255)
            extra = dict(psnr_s1=float(calculate_psnr(o1, g)), ssim_s1=float(calculate_ssim(o1, g)),
                         gt_psnr_s1=float(calculate_psnr(og1, g)), bright_s1=float(m1 / mg))
        rows[stem] = dict(psnr=float(calculate_psnr(o, g)), ssim=float(calculate_ssim(o, g)),
                          gt_psnr=float(calculate_psnr(og, g)), gt_ssim=float(calculate_ssim(og, g)),
                          bright=float(mo / mg), lpips=lpv, **extra)
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("psnr", "ssim", "gt_psnr", "gt_ssim", "bright", "lpips", "psnr_a1", "psnr_s1", "ssim_s1", "gt_psnr_s1", "bright_s1") if all(k in r for r in rows.values())}
    out_dir = os.path.join("analysis", "per_image")
    os.makedirs(out_dir, exist_ok=True)
    ck = os.path.splitext(os.path.basename(wpath))[0] if a.ckpt != "best" else "best"
    json.dump(dict(tag=a.tag, weights=wpath, mean=mean, images=rows), open(os.path.join(out_dir, f"{a.tag}_{ck}.json"), "w"), indent=1)
    print(a.tag, ck, json.dumps(mean))


if __name__ == "__main__":
    main()
