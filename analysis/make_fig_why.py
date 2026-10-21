# -*- coding: utf-8 -*-
"""Single-column mechanistic-diagnosis figure (Fig. 3).

(a) H1  test-set R^2 of ridge probes fitted on training inputs, decoding log input intensity from F_I, LN(F_I), and
        the per-pixel mean/std that LN removes; IG-LCA official, 3 seeds (dots).
(b) H2a channel-wise std (std over channels of each channel's spatial mean of G) and spatial std (std over pixels of
        the channel-averaged gate map), averaged over test images; 7 trained gates (I_LCA5 is overwritten), 3 seeds.
(c) H2b inference-time intervention without retraining: GT-mean PSNR change relative to the learned gate when G is
        fixed to its per-channel mean over the TRAINING images, or to 1; per seed.
(d) H3  GT-mean PSNR change versus matched CIDNet for IG-LCA and the explicit-intensity variants; one-seed variants
        are exploratory and drawn hollow and hatched.
Error bars: within-seed 95% image-bootstrap intervals (the paper's primary cross-seed inference is hierarchical).
Inputs: analysis/ln_probe.json, analysis/gate_stats_<tag>.json, analysis/gate_static_train_gt_<tag>.json,
        analysis/per_image/*official*_epoch_1000.json.
Usage: python analysis/make_fig_why.py <out.pdf>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 6.5, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "hatch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})
SEEDS = ["3407", "42", "1234"]
BLUE, ORANGE, GRAY, GREEN, RED, PURPLE = "#2f6db3", "#d9822b", "#9a9a9a", "#2e8b57", "#c0392b", "#7f5fa8"
LIGHT = {"H1": "#dce8f6", "H2": "#dcefe2", "H3": "#f6dddb"}
DARK = {"H1": BLUE, "H2": GREEN, "H3": RED}
PI = os.path.join("analysis", "per_image")
TAG = "lolv2_real_ablation_iglca_3456_official_s{}"


def boot_mean_ci(d, B=4000, seed=0):
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def head(ax, color_key, text, verdict_text):
    ax.text(-0.02, 1.05, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.9, fontweight="bold",
            color=DARK[color_key], linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.2", fc=LIGHT[color_key], ec="none"))
    ax.text(1.0, 1.05, verdict_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.22", fc=DARK[color_key], ec="none"))


def load_pi(name):
    f = os.path.join(PI, f"{name}_epoch_1000.json")
    return json.load(open(f, encoding="utf-8"))["images"] if os.path.exists(f) else None


def main():
    out = sys.argv[1]
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fig, axs = plt.subplots(2, 2, figsize=(3.45, 2.6))
    fig.subplots_adjust(left=0.145, right=0.985, top=0.88, bottom=0.108, wspace=0.40, hspace=0.98)

    # (a) H1: information absent?
    ax = axs[0, 0]
    P = json.load(open(os.path.join("analysis", "ln_probe.json")))
    blocks = ["LCA3", "LCA4", "LCA5", "LCA6"]
    for j, (key, col, mk, lab) in enumerate((("pre", GRAY, "o", r"$F_{\mathrm{I}}$"), ("post", BLUE, "s", r"LN$(F_{\mathrm{I}})$"),
                                              ("removed", ORANGE, "^", r"removed $\mu,\sigma$"))):
        for i, b in enumerate(blocks):
            v = [P[f"iglca_3456_s{s}"][b][key]["r2"] for s in SEEDS]
            x = i + (j - 1) * 0.22
            ax.scatter([x] * len(v), v, s=6, color=col, marker=mk, lw=0, zorder=3, label=lab if i == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels(blocks)
    ax.set_xlim(-0.55, 3.5)
    ax.set_ylim(-0.2, 1.06)
    ax.set_ylabel(r"probe $R^2$ (test)", labelpad=1)
    ax.legend(fontsize=5.0, frameon=False, loc="lower left", handletextpad=0.05, borderaxespad=0.15,
              labelspacing=0.12, borderpad=0.1)
    head(ax, "H1", "H1: Information\nabsent?", "Rejected")

    # (b) H2a: weak input dependence?
    ax = axs[0, 1]
    gates = ["HV_LCA3", "HV_LCA4", "HV_LCA5", "HV_LCA6", "I_LCA3", "I_LCA4", "I_LCA6"]  # I_LCA5 is overwritten
    names = ["HV3", "HV4", "HV5", "HV6", "I3", "I4", "I6"]
    for k, (key, col, lab) in enumerate((("ch_std", GREEN, "channel-wise std."), ("px_std", GRAY, "spatial std."))):
        for i, g in enumerate(gates):
            v = [json.load(open(os.path.join("analysis", f"gate_stats_{TAG.format(s)}.json")))[g][key] for s in SEEDS]
            x = i + (k - 0.5) * 0.3
            ax.bar(x, np.mean(v), width=0.3, color=col, alpha=0.85, lw=0, label=lab if i == 0 else None, zorder=2)
            ax.scatter([x] * 3, v, s=2.5, color="black", lw=0, zorder=3)
    ax.set_xticks(range(len(gates)))
    ax.set_xticklabels(names, fontsize=5.6)
    ax.set_ylabel("std of gate $G$", labelpad=1)
    ax.set_ylim(0, 0.34)
    ax.legend(fontsize=5.0, frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.1, labelspacing=0.15,
              handlelength=1.0)
    head(ax, "H2", "H2a: Weak input\ndependence?", "Supported")

    # (c) H2b: fixed-gate intervention (inference only), GT-mean PSNR change relative to the learned gate
    ax = axs[1, 0]
    for k, (mode, lab, col) in enumerate((("static_train", r"$G\leftarrow\bar G_{\mathrm{train}}$", GREEN),
                                           ("one", r"$G\leftarrow 1$", GRAY))):
        for i, s in enumerate(SEEDS):
            fj = os.path.join("analysis", f"gate_static_train_gt_{TAG.format(s)}.json")
            if not os.path.exists(fj):  # still running: leave the bar out
                continue
            J = json.load(open(fj))
            d = np.array(J[mode]["per_image_gt_psnr"]) - np.array(J["learned"]["per_image_gt_psnr"])
            m, lo, hi = boot_mean_ci(d)
            x = i + (k - 0.5) * 0.3
            ax.bar(x, m, width=0.3, color=col, alpha=0.85, lw=0, label=lab if i == 0 else None, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", lw=0.5, capsize=1.2, zorder=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(SEEDS)
    ax.set_xlabel("seed", labelpad=1)
    # two lines: the one-line label ran into the panel letter above the axes
    ax.set_ylabel("GT-mean\nchange (dB)", labelpad=1, linespacing=0.95)
    ax.set_ylim(-0.75, 0.75)
    ax.legend(fontsize=5.0, frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.1, labelspacing=0.15,
              handlelength=1.0)
    head(ax, "H2", "H2b: Fixed-gate\nintervention", "Small effect")

    # (d) H3: does explicit intensity help? one-seed variants hollow and hatched (exploratory)
    ax = axs[1, 1]
    variants = [("iglca_3456", "IG-LCA", BLUE, False), ("iglca_3456_iig", "intensity", ORANGE, False),
                ("iglca_3456_iigf", "feat.+int.*", GRAY, True), ("iglca_3456_iige", "int.+global*", PURPLE, True)]
    w = 0.2
    for k, (model, lab, col, one_seed) in enumerate(variants):
        for i, s in enumerate(SEEDS):
            A, Bm = load_pi(f"lolv2_real_ablation_lca_none_official_s{s}"), load_pi(f"lolv2_real_ablation_{model}_official_s{s}")
            if A is None or Bm is None:
                continue
            d = [Bm[n]["gt_psnr"] - A[n]["gt_psnr"] for n in sorted(A)]
            m, lo, hi = boot_mean_ci(d)
            x = i + (k - 1.5) * w
            style = dict(facecolor="white", edgecolor=col, hatch="////", lw=0.6) if one_seed else dict(color=col, alpha=0.85, lw=0)
            ax.bar(x, m, width=w, label=lab if i == 0 else None, zorder=2, **style)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", lw=0.5, capsize=1.0, zorder=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(SEEDS)
    ax.set_xlabel("seed", labelpad=1)
    # two lines: the one-line label ran into the panel letter above the axes
    ax.set_ylabel("GT-mean\nchange (dB)", labelpad=1, linespacing=0.95)
    ax.set_ylim(-0.75, 0.75)
    ax.legend(fontsize=4.9, frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.1, labelspacing=0.12,
              handlelength=1.0, ncol=2, columnspacing=0.6)
    head(ax, "H3", "H3: Does explicit\nintensity help?", "Not supported")

    for a, t in zip(axs.flat, "abcd"):
        a.text(-0.32, 1.05, f"({t})", transform=a.transAxes, fontsize=6.4, fontweight="bold", va="bottom")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
