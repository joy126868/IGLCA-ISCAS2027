# -*- coding: utf-8 -*-
"""Full-width figure for the direction-B paper (four panels).

(a) Gain over CIDNet (same loss and seed): PSNR (x) vs GT-mean PSNR (y). Blue: IG-LCA-3456 perceptual-only (one marker
    per seed); orange: IG-LCA-3456 official loss (one marker per seed). The gate placements are in the architecture table.
(b) Per image, IG-LCA-3456 vs CIDNet, perceptual-only, all available seeds: change of brightness error vs change of PSNR.
(c) LOLv2-Real ground-truth mean gray level, training vs test images (histograms).
(d) PSNR vs output scale alpha on test images (solid) and on every 7th training image (dashed) for the released CIDNet
    weights and our CIDNet / IG-LCA (official loss, seed 3407); markers at each curve's maximum.
Usage: python analysis/make_fig_b.py <out.pdf>
Inputs: analysis/per_image/*.json, analysis/alpha_sweep.json, analysis/dataset_brightness.json (make_numbers_b.py).
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
PI = os.path.join("analysis", "per_image")
SEEDS = ["3407", "42", "1234"]
BLUE, ORANGE, GRAY = "#2f6db3", "#d9822b", "#9a9a9a"


def load(tag):
    f = os.path.join(PI, f"{tag}_epoch_1000.json")
    return json.load(open(f, encoding="utf-8"))["images"] if os.path.exists(f) else None


def gain(A, B):
    s = sorted(A)
    return np.mean([B[k]["psnr"] - A[k]["psnr"] for k in s]), np.mean([B[k]["gt_psnr"] - A[k]["gt_psnr"] for k in s])


def main():
    out = sys.argv[1]
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.62), gridspec_kw={"width_ratios": [1, 1, 0.9, 1.1]})

    # (a) gains
    a = ax[0]
    base_p = load("lolv2_real_ablation_lca_none_percep_s3407")
    # so panel (a) shows only the matched-seed IG-LCA-3456 comparisons
    _ = base_p
    for rec, col, mk, lab in (("percep", BLUE, "o", "IG-LCA-3456, perc."), ("official", ORANGE, "s", "IG-LCA-3456, official")):
        g = []
        for s in SEEDS:
            A, B = load(f"lolv2_real_ablation_lca_none_{rec}_s{s}"), load(f"lolv2_real_ablation_iglca_3456_{rec}_s{s}")
            if A is not None and B is not None:
                g.append(gain(A, B))
        if g:
            g = np.array(g)
            a.scatter(g[:, 0], g[:, 1], s=13, color=col, marker=mk, lw=0, label=lab, zorder=3)
    # the limits now cover every plotted point.
    lim = (-0.9, 1.6)
    a.plot(lim, lim, color="#bbbbbb", lw=0.5, ls="--", zorder=1)
    a.axhline(0, color="black", lw=0.5)
    a.axvline(0, color="black", lw=0.5)
    a.set_xlim(lim)
    a.set_ylim(-0.6, 1.1)
    a.set_xlabel("PSNR gain (dB)", labelpad=1)
    a.set_ylabel("GT-mean PSNR gain (dB)", labelpad=1)
    a.legend(fontsize=4.8, frameon=False, loc="lower right", handletextpad=0.1, borderaxespad=0.05, labelspacing=0.2)
    a.set_title("(a) gain over CIDNet", fontsize=7, pad=2)

    # (b) per image, perceptual-only
    b = ax[1]
    de, dp = [], []
    for s in SEEDS:
        A, B = load(f"lolv2_real_ablation_lca_none_percep_s{s}"), load(f"lolv2_real_ablation_iglca_3456_percep_s{s}")
        if A is None or B is None:
            continue
        for k in A:
            de.append(abs(np.log(B[k]["bright"])) - abs(np.log(A[k]["bright"])))
            dp.append(B[k]["psnr"] - A[k]["psnr"])
    b.scatter(de, dp, s=3, color=BLUE, lw=0, alpha=0.8)
    b.axhline(0, color="black", lw=0.5)
    b.axvline(0, color="black", lw=0.5)
    r_b = float(np.corrcoef(de, dp)[0, 1])
    b.text(0.97, 0.95, f"$r={r_b:.2f}$\n$n={len(de)}$", transform=b.transAxes, ha="right", va="top", fontsize=6)
    b.set_xlabel("change of brightness error", labelpad=1)
    b.set_ylabel("PSNR change per image (dB)", labelpad=1)
    b.set_title("(b) IG-LCA vs CIDNet, perc.", fontsize=7, pad=2)

    # (c) dataset brightness: input vs target mean gray level, training and test pairs
    c = ax[2]
    D = json.load(open(os.path.join("analysis", "dataset_brightness.json")))
    tr, te = np.array(D["Train"]["gt"]), np.array(D["Test"]["gt"])
    ltr, lte = np.array(D["Train"]["low"]), np.array(D["Test"]["low"])
    c.scatter(ltr, tr, s=2.5, color=GRAY, lw=0, alpha=0.6, label="train", zorder=2)
    c.scatter(lte, te, s=5, color=ORANGE, lw=0, alpha=0.9, label="test", zorder=3)
    c.set_xscale("log")
    c.set_yscale("log")
    c.axhline(tr.mean(), color="#555555", lw=0.6, ls="--", zorder=1)
    c.axhline(te.mean(), color=ORANGE, lw=0.6, ls="--", zorder=1)
    c.set_xlim(0.25, 110)
    c.text(0.3, tr.mean() * 1.04, f"train mean {tr.mean():.0f}", fontsize=5, color="#444444", ha="left", va="bottom")
    c.text(0.3, te.mean() * 0.96, f"test mean {te.mean():.0f}", fontsize=5, color=ORANGE, ha="left", va="top")
    c.set_xlabel("input mean gray level", labelpad=1)
    c.set_ylabel("target mean gray level", labelpad=1)
    c.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    c.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    c.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    c.set_yticks([40, 70, 100, 150, 200])
    c.set_xticks([1, 3, 10, 30, 100])
    c.legend(fontsize=4.8, frameon=False, loc="upper left", handletextpad=0.1, borderaxespad=0.05, markerscale=1.8)
    c.set_title("(c) LOLv2-Real pairs", fontsize=7, pad=2)

    # (d) alpha sweep
    d = ax[3]
    A = json.load(open(os.path.join("analysis", "alpha_sweep.json")))
    show = [("released_bestPSNR", "#444444", "released"), ("base_official_s3407", BLUE, "CIDNet (ours)"),
            ("IG_official_s3407", ORANGE, "IG-LCA")]
    for key, col, lab in show:
        if key not in A:
            continue
        al = np.array(A[key]["alphas"])
        for split, ls in (("test", "-"), ("train", "--")):
            y = np.array(A[key][split])
            d.plot(al, y, color=col, lw=0.9, ls=ls, label=lab if split == "test" else None)
            j = int(y.argmax())
            d.plot(al[j], y[j], marker="o", ms=2.2, color=col)
    d.axvline(0.8, color="#999999", lw=0.5, ls=":")
    d.text(0.805, d.get_ylim()[1] - 0.2, r"$\alpha$=0.8", fontsize=5.5, color="#666666", va="top")
    d.set_xlabel(r"output scale $\alpha$", labelpad=1)
    d.set_ylabel("PSNR (dB)", labelpad=1)
    d.legend(fontsize=4.8, frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.05, ncol=1,
             title="solid: test, dashed: train", title_fontsize=4.8)
    d.set_title(r"(d) PSNR vs $\alpha$", fontsize=7, pad=2)

    for x in ax:
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        x.tick_params(length=2, pad=1, labelsize=6)
    fig.tight_layout(pad=0.2, w_pad=0.7)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
