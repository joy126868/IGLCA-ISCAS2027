# -*- coding: utf-8 -*-
"""Per-image scores of a Retinexformer ablation checkpoint (train_rf_ablation.py), in the format of eval_run.py.

Tag: lolv2_real_ablation_rf_<ig|noig>_official_s<seed>. The network is built from the patched copy
(third_party/Retinexformer_iscas, ig switch) and run as in the official Enhancement/test_from_dataset.py (input padded
to a multiple of 4 by reflection, no output scale, no GT-mean rectification). Scores are computed exactly as for the
CIDNet runs (8-bit RGB PSNR/SSIM, GT-mean PSNR, AlexNet LPIPS, brightness ratio), so analysis/pair_report.py can compare
rf_noig with rf_ig:  python analysis/pair_report.py lolv2_real rf_noig rf_ig --ckpt iter_150000
Usage: python analysis/eval_rf.py <tag> [--iter 150000] [--limit N] [--exp-suffix _debug]
Writes analysis/per_image/<tag>_iter_<iter>.json.
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from measure import calculate_psnr, calculate_ssim  # noqa: E402

RF = os.path.join(ROOT, "third_party", "Retinexformer_iscas")
ARCH = os.path.join(RF, "basicsr", "models", "archs", "RetinexFormer_arch.py")
LOW, GT = "LOLV2/LOL-v2/Real_captured/Test/Low", "LOLV2/LOL-v2/Real_captured/Test/Normal"


def build(ig):
    spec = importlib.util.spec_from_file_location("rf_arch_iscas", ARCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # network_g of the official LOLv2-Real option file
    return mod.RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2], ig=ig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--iter", default="150000", help="iteration number or latest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exp-suffix", default="", help="e.g. _debug for the pipeline smoke test")
    a = ap.parse_args()
    m = re.match(r"lolv2_real_ablation_rf_((?:no)?ig\d*)_official_s(\d+)$", a.tag)
    if not m:
        raise SystemExit(f"not a Retinexformer ablation tag: {a.tag}")
    variant, seed = m.group(1), m.group(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wpath = os.path.join(RF, "experiments", f"RF_LOLv2real_{variant}_s{seed}{a.exp_suffix}", "models", f"net_g_{a.iter}.pth")
    net = build(variant.startswith("ig")).to(dev).eval()  # ig128 -> guided, noig128 -> unguided
    ck = torch.load(wpath, map_location="cpu")
    net.load_state_dict(ck["params"] if "params" in ck else ck)
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    files = sorted(glob.glob(os.path.join(LOW, "*.png")))
    if a.limit:
        files = files[:a.limit]
    rows = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        g = np.asarray(Image.open(os.path.join(GT, stem.replace("low", "normal") + ".png")).convert("RGB"))
        x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        h, w = x.shape[2:]
        ph, pw = (-h) % 4, (-w) % 4
        with torch.no_grad():
            y = net(F.pad(x, (0, pw, 0, ph), "reflect") if (ph or pw) else x)[:, :, :h, :w].clamp(0, 1)
        o = (y[0].permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
        mo = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).mean()
        mg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY).mean()
        og = np.clip(o.astype(np.float64) * (mg / max(mo, 1e-6)), 0, 255)
        with torch.no_grad():
            lpv = float(lp(lpips.im2tensor(g).to(dev), lpips.im2tensor(o).to(dev)).item())
        rows[stem] = dict(psnr=float(calculate_psnr(o, g)), ssim=float(calculate_ssim(o, g)),
                          gt_psnr=float(calculate_psnr(og, g)), gt_ssim=float(calculate_ssim(og, g)),
                          bright=float(mo / mg), lpips=lpv)
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("psnr", "ssim", "gt_psnr", "gt_ssim", "bright", "lpips")}
    out = os.path.join("analysis", "per_image", f"{a.tag}_iter_{a.iter}.json")
    if a.exp_suffix:  # smoke tests never overwrite real results
        out = out.replace(".json", f"{a.exp_suffix}.json")
    json.dump(dict(tag=a.tag, weights=wpath, mean=mean, images=rows), open(out, "w"), indent=1)
    print(a.tag, a.iter, json.dumps({k: round(v, 4) for k, v in mean.items()}), "->", out)


if __name__ == "__main__":
    main()
