# -*- coding: utf-8 -*-
"""Mechanism analysis of IG-LCA (reviewer #27): what do the learned gates do?

For every IlluminationGate in a trained IG-LCA-3456 model, run the LOLv2-real test set (same eval
settings as training: gated2 decode, alpha 0.8) and record the gate G = 2*sigmoid(g(guide)):
  - mean, std, mean |G-1|, share of elements with |G-1| > 0.1   (how far from the plain LCA, G = 1)
  - spread across channels and across pixels
  - relation to brightness: Pearson correlation between the channel-mean gate map and the log of the
    input brightness max(R,G,B) at the gate's resolution, and the mean gate in the darkest vs brightest
    fifth of pixels of each image.
The I_LCA5 gate is a sanity check: the official CIDNet forward overwrites I_LCA5's output, so its gate
never receives a gradient and must stay exactly 1.

Usage: python analysis/gate_stats.py <weights.pth> <out.json> [--threads 4] [--limit N]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from net.CIDNet_IGLCA_ABLATION_3456 import CIDNet  # noqa: E402
from net.IG_LCA import IlluminationGate  # noqa: E402

TEST_LOW = "LOLV2/LOL-v2/Real_captured/Test/Low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    model = CIDNet()
    state = torch.load(a.weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.trans.gated2 = True
    model.trans.alpha = 0.8

    gates = {n.replace(".illum_gate", ""): m for n, m in model.named_modules() if isinstance(m, IlluminationGate)}
    cur = {}
    for name, m in gates.items():
        m.register_forward_hook(lambda mod, inp, out, name=name: cur.__setitem__(name, out.detach()))

    files = sorted(glob.glob(os.path.join(TEST_LOW, "*.png")))
    if a.limit:
        files = files[:a.limit]
    acc = {n: dict(mean=[], std=[], dev=[], frac=[], ch_std=[], px_std=[], corr=[], dark=[], bright=[]) for n in gates}
    for k, f in enumerate(files):
        x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        cur.clear()
        with torch.no_grad():
            model(x)
        v = x.max(dim=1, keepdim=True).values  # HSV value = HVI intensity of the input
        for n, g in cur.items():
            g = g[0]  # C x h x w
            s = acc[n]
            s["mean"].append(g.mean().item())
            s["std"].append(g.std().item())
            s["dev"].append((g - 1).abs().mean().item())
            s["frac"].append(((g - 1).abs() > 0.1).float().mean().item())
            s["ch_std"].append(g.mean(dim=(1, 2)).std().item())       # spread of per-channel means
            gm = g.mean(dim=0)                                          # h x w, channel-mean gate map
            s["px_std"].append(gm.std().item())
            b = F.adaptive_avg_pool2d(v, gm.shape)[0, 0]
            lb = torch.log(b + 1e-3).flatten()
            gg = gm.flatten()
            if gg.std() > 1e-8 and lb.std() > 1e-8:
                s["corr"].append(torch.corrcoef(torch.stack([gg, lb]))[0, 1].item())
            q = torch.quantile(b.flatten(), torch.tensor([0.2, 0.8]))
            s["dark"].append(gg[b.flatten() <= q[0]].mean().item())
            s["bright"].append(gg[b.flatten() >= q[1]].mean().item())
        if (k + 1) % 20 == 0:
            print(f"{k + 1}/{len(files)}", flush=True)

    out = {}
    for n, s in acc.items():
        out[n] = {key: (float(np.mean(val)) if val else None) for key, val in s.items()}
        out[n]["corr_n"] = len(s["corr"])
        out[n]["shape_c"] = int(gates[n].body[-1].out_channels)
    out["_meta"] = dict(weights=a.weights, images=len(files), decode="gated2, alpha 0.8")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"{'gate':<10}{'C':>4}{'mean G':>8}{'std':>7}{'|G-1|':>7}{'>0.1':>6}{'ch-std':>7}{'px-std':>7}{'corr(logV)':>11}{'dark':>7}{'bright':>7}")
    for n in sorted(gates, key=lambda s: (s[:2], s)):
        o = out[n]
        print(f"{n:<10}{o['shape_c']:>4}{o['mean']:>8.3f}{o['std']:>7.3f}{o['dev']:>7.3f}{o['frac']:>6.2f}{o['ch_std']:>7.3f}"
              f"{o['px_std']:>7.3f}{(o['corr'] if o['corr'] is not None else float('nan')):>11.3f}{o['dark']:>7.3f}{o['bright']:>7.3f}")


if __name__ == "__main__":
    main()
