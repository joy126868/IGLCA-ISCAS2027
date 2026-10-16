# -*- coding: utf-8 -*-
"""Numbers and the main table for the direction-B paper (analysis: the gate's gain is brightness, not structure).

Usage: python analysis/make_numbers_b.py <paper folder>
Writes <paper>/numbers_b.tex (macros, prefix \\B...) and <paper>/tab_main_b.tex. Missing values print as ??.
Inputs (all produced by other scripts in analysis/):
  per_image/<ds>_ablation_<model>_<recipe>_s<seed>_epoch_1000.json   eval_run.py (psnr, psnr_a1, gt_psnr, ssim, lpips, bright)
  alpha_sweep.json                                                   alpha_sweep.py (PSNR vs output scale alpha)
  gate_stats_<tag>.json, gate_intervene_<tag>.json                    gate_stats.py, gate_intervene.py (official IG runs)
  released_eval/HVI-CIDNet-LOLv2-real-bestPSNR_a0.8/scores.json       eval_released.py
  dataset_brightness.json                                            computed here once (LOLv2-Real train/test means)
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import stats

SEEDS = ["3407", "42", "1234"]
PI = os.path.join("analysis", "per_image")
NA = r"\TBD{}"  # red [TBD] in the paper (defined in main_iscas.tex): result still missing


def pi(model, recipe, seed, ds="lolv2_real"):
    f = os.path.join(PI, f"{ds}_ablation_{model}_{recipe}_s{seed}_epoch_1000.json")
    return json.load(open(f, encoding="utf-8"))["images"] if os.path.exists(f) else None


def berr(rows):
    return {k: abs(np.log(v["bright"])) for k, v in rows.items()}


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return NA
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    # \ensuremath: the macros are used both in text and inside $...$
    return f"\\ensuremath{{{m:.0f}{{\\times}}10^{{{e}}}}}" if round(m) != 10 else f"\\ensuremath{{10^{{{e + 1}}}}}"


def sgn(x, nd=2):
    return NA if x is None or not np.isfinite(x) else (f"+{x:.{nd}f}" if x >= 0 else f"\\ensuremath{{-}}{abs(x):.{nd}f}")


def collect(recipe):
    """Per-seed means and pooled paired differences (IG - baseline) over the seeds where both runs exist."""
    seeds = [s for s in SEEDS if pi("lca_none", recipe, s) and pi("iglca_3456", recipe, s)]
    keys = ["psnr", "psnr_a1", "gt_psnr", "ssim", "lpips", "berr"]
    out = dict(seeds=seeds, base={k: [] for k in keys}, ig={k: [] for k in keys}, d={k: [] for k in keys})
    for s in seeds:
        A, B = pi("lca_none", recipe, s), pi("iglca_3456", recipe, s)
        ea, eb = berr(A), berr(B)
        for k in keys:
            if k == "berr":
                a = np.array([ea[i] for i in sorted(A)])
                b = np.array([eb[i] for i in sorted(A)])
            elif all(k in A[i] and k in B[i] for i in A):
                a = np.array([A[i][k] for i in sorted(A)])
                b = np.array([B[i][k] for i in sorted(A)])
            else:
                continue
            out["base"][k].append(a.mean())
            out["ig"][k].append(b.mean())
            out["d"][k].extend((b - a).tolist())
    out["p"] = {k: (stats.wilcoxon(v).pvalue if len(v) and np.any(np.array(v) != 0) else None) for k, v in out["d"].items()}
    out["dm"] = {k: (float(np.mean(v)) if len(v) else None) for k, v in out["d"].items()}
    return out


def ms(v, nd=2):
    if not v:
        return NA
    return f"{np.mean(v):.{nd}f}" if len(v) == 1 else f"{np.mean(v):.{nd}f}\\pmr{{{np.std(v):.{nd}f}}}"


def dataset_brightness():
    f = os.path.join("analysis", "dataset_brightness.json")
    if os.path.exists(f):
        return json.load(open(f))
    res = {}
    for split in ("Train", "Test"):
        lo, gt = [], []
        for p in sorted(glob.glob(f"LOLV2/LOL-v2/Real_captured/{split}/Low/*.png")):
            g = p.replace(os.sep + "Low" + os.sep, os.sep + "Normal" + os.sep).replace("/Low/", "/Normal/")
            g = os.path.join(os.path.dirname(os.path.dirname(p)), "Normal", os.path.basename(p).replace("low", "normal"))
            lo.append(float(cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2GRAY).mean()))
            gt.append(float(cv2.cvtColor(np.asarray(Image.open(g).convert("RGB")), cv2.COLOR_RGB2GRAY).mean()))
        res[split] = dict(low=lo, gt=gt)
    json.dump(res, open(f, "w"))
    return res


def dataset_saturation_v1():
    """Mean HSV saturation (OpenCV, 0-255) of the LOLv1 targets: the colour counterpart of the LOLv2-Real shift."""
    f = os.path.join("analysis", "dataset_saturation_lolv1.json")
    if os.path.exists(f):
        return json.load(open(f))
    sat = lambda p: float(cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2HSV)[..., 1].mean())
    res = {split: [sat(p) for p in sorted(glob.glob(os.path.join(d, "*.png")))]
           for split, d in (("Train", "LOL_v1/train/high"), ("Test", "LOL_v1/test/normal"))}
    json.dump(res, open(f, "w"))
    return res


def dataset_brightness_v1():
    """Mean gray levels of the LOLv1 pairs (train: low/high, test: low/normal), the control for the LOLv2-Real shift."""
    f = os.path.join("analysis", "dataset_brightness_lolv1.json")
    if os.path.exists(f):
        return json.load(open(f))
    gray = lambda p: float(cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2GRAY).mean())
    res = {}
    for split, lo_dir, gt_dir in (("Train", "LOL_v1/train/low", "LOL_v1/train/high"), ("Test", "LOL_v1/test/low", "LOL_v1/test/normal")):
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(lo_dir, "*.png")))
        res[split] = dict(low=[gray(os.path.join(lo_dir, n)) for n in names], gt=[gray(os.path.join(gt_dir, n)) for n in names])
    json.dump(res, open(f, "w"))
    return res


def main():
    paper = sys.argv[1]
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    M = {}
    P, O = collect("percep"), collect("official")
    for tag, R in (("P", P), ("O", O)):
        M[f"B{tag}N"] = str(len(R["seeds"]))
        M[f"B{tag}Base"] = f"{np.mean(R['base']['psnr']):.2f}" if R["seeds"] else NA
        M[f"B{tag}IG"] = f"{np.mean(R['ig']['psnr']):.2f}" if R["seeds"] else NA
        M[f"B{tag}Gain"] = sgn(R["dm"]["psnr"])
        M[f"B{tag}GainGT"] = sgn(R["dm"]["gt_psnr"])
        M[f"B{tag}GainAone"] = sgn(R["dm"]["psnr_a1"])
        M[f"B{tag}GainSSIM"] = sgn(R["dm"]["ssim"], 3)
        M[f"B{tag}GainLPIPS"] = sgn(R["dm"]["lpips"], 3)
        M[f"B{tag}P"] = fmt_p(R["p"]["psnr"])
        M[f"B{tag}PGT"] = fmt_p(R["p"]["gt_psnr"])
        M[f"B{tag}PAone"] = fmt_p(R["p"]["psnr_a1"])
        g, gt = R["dm"]["psnr"], R["dm"]["gt_psnr"]
        M[f"B{tag}Share"] = f"{100 * (g - gt) / g:.0f}" if g and gt is not None and abs(g) > 1e-6 else NA
        M[f"B{tag}BerrBase"] = f"{np.mean(R['base']['berr']):.3f}" if R["seeds"] else NA
        M[f"B{tag}BerrIG"] = f"{np.mean(R['ig']['berr']):.3f}" if R["seeds"] else NA
        # per-seed PSNR difference and its brightness part (PSNR difference minus GT-mean difference)
        rec_ = "percep" if tag == "P" else "official"
        sd, sb, sa = [], [], []
        for s in R["seeds"]:
            A_, B_ = pi("lca_none", rec_, s), pi("iglca_3456", rec_, s)
            dps = float(np.mean([B_[i]["psnr"] - A_[i]["psnr"] for i in A_]))
            dgt = float(np.mean([B_[i]["gt_psnr"] - A_[i]["gt_psnr"] for i in A_]))
            sd.append(dps)
            sb.append(dps - dgt)
            if all("psnr_a1" in A_[i] and "psnr_a1" in B_[i] for i in A_):
                sa.append(float(np.mean([B_[i]["psnr_a1"] - A_[i]["psnr_a1"] for i in A_])))

        def lst(v):
            v = [sgn(x) for x in v]
            return NA if not v else (v[0] if len(v) == 1 else ", ".join(v[:-1]) + (", and " if len(v) > 2 else " and ") + v[-1])
        M[f"B{tag}SeedGains"] = lst(sd)
        M[f"B{tag}SeedBright"] = lst(sb)
        sbe_b, sbe_g = [], []
        for s in R["seeds"]:
            A_, B_ = pi("lca_none", rec_, s), pi("iglca_3456", rec_, s)
            sbe_b.append(float(np.mean([abs(np.log(A_[i]["bright"])) for i in A_])))
            sbe_g.append(float(np.mean([abs(np.log(B_[i]["bright"])) for i in B_])))
        fmt3 = lambda v: NA if not v else (f"{v[0]:.3f}" if len(v) == 1 else ", ".join(f"{x:.3f}" for x in v[:-1]) + (", and " if len(v) > 2 else " and ") + f"{v[-1]:.3f}")
        M[f"B{tag}SeedBerrBase"] = fmt3(sbe_b)
        M[f"B{tag}SeedBerrIG"] = fmt3(sbe_g)
        # per-seed GT-mean PSNR of each model (how much the runs themselves vary)
        fmt2 = lambda v: NA if not v else (f"{v[0]:.2f}" if len(v) == 1 else ", ".join(f"{x:.2f}" for x in v[:-1]) + (", and " if len(v) > 2 else " and ") + f"{v[-1]:.2f}")
        gb = [float(np.mean([pi("lca_none", rec_, s)[i]["gt_psnr"] for i in pi("lca_none", rec_, s)])) for s in R["seeds"]]
        gi = [float(np.mean([pi("iglca_3456", rec_, s)[i]["gt_psnr"] for i in pi("iglca_3456", rec_, s)])) for s in R["seeds"]]
        M[f"B{tag}SeedGTBase"], M[f"B{tag}SeedGTIG"] = fmt2(gb), fmt2(gi)
        M[f"B{tag}NWord"] = {1: "one seed", 2: "two seeds", 3: "three seeds"}.get(len(R["seeds"]), f"{len(R['seeds'])} seeds")
        M[f"B{tag}SeedGainsAone"] = lst(sa) if len(sa) == len(sd) else NA
        for who in ("base", "ig"):
            for k, nm in (("psnr", ""), ("gt_psnr", "GT")):
                v = R[who][k]
                M[f"B{tag}Range{'Base' if who == 'base' else 'IG'}{nm}"] = f"{max(v) - min(v):.2f}" if len(v) > 1 else NA
        if R["d"]["psnr"]:
            # per-image correlation between the change of brightness error and the change of PSNR (pooled)
            dp, de = [], []
            for s in R["seeds"]:
                rec = "percep" if tag == "P" else "official"
                A, B = pi("lca_none", rec, s), pi("iglca_3456", rec, s)
                ea, eb = berr(A), berr(B)
                for i in A:
                    dp.append(B[i]["psnr"] - A[i]["psnr"])
                    de.append(eb[i] - ea[i])
            M[f"B{tag}Corr"] = sgn(stats.pearsonr(de, dp)[0])
        else:
            M[f"B{tag}Corr"] = NA

    # ---- hierarchical bootstrap: resample seeds, then images within each seed (images repeat across seeds)
    def hboot(rec, key, B=20000):
        seeds = [s for s in SEEDS if pi("lca_none", rec, s) and pi("iglca_3456", rec, s)]
        if not seeds:
            return NA
        rng = np.random.default_rng(0)
        D = []
        for s in seeds:
            A_, B_ = pi("lca_none", rec, s), pi("iglca_3456", rec, s)
            D.append(np.array([B_[i][key] - A_[i][key] for i in sorted(A_)]))
        Mimg = np.stack([d[rng.integers(0, len(d), (B, len(d)))].mean(1) for d in D], axis=1)
        pick = rng.integers(0, len(D), (B, len(D)))
        boot = np.take_along_axis(Mimg, pick, axis=1).mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return f"[{sgn(float(lo))}, {sgn(float(hi))}]"
    for tag, rec in (("P", "percep"), ("O", "official")):
        M[f"B{tag}CIPsnr"] = hboot(rec, "psnr")
        M[f"B{tag}CIGT"] = hboot(rec, "gt_psnr")

    # ---- LOLv1, official loss (second dataset; 15 test images, gated decoding as in the official code)
    def boot_ci(per_seed, B=20000):
        rng = np.random.default_rng(0)
        D = [np.asarray(d) for d in per_seed]
        Mimg = np.stack([d[rng.integers(0, len(d), (B, len(d)))].mean(1) for d in D], axis=1)
        pick = rng.integers(0, len(D), (B, len(D)))
        lo, hi = np.percentile(np.take_along_axis(Mimg, pick, axis=1).mean(1), [2.5, 97.5])
        return f"[{sgn(float(lo))}, {sgn(float(hi))}]"
    seedsL = [s for s in SEEDS if pi("lca_none", "official", s, ds="lol_v1") and pi("iglca_3456", "official", s, ds="lol_v1")]
    M["BLN"] = str(len(seedsL))
    M["BLNWord"] = {0: "no seeds", 1: "one seed", 2: "two seeds", 3: "three seeds"}.get(len(seedsL), f"{len(seedsL)} seeds")
    keysL = ("psnr", "gt_psnr", "ssim", "lpips")
    if seedsL:
        pooled = {k: [] for k in keysL}
        per_seed = {k: [] for k in keysL}
        base_m = {k: [] for k in keysL}
        ig_m = {k: [] for k in keysL}
        for s in seedsL:
            A_, B_ = pi("lca_none", "official", s, ds="lol_v1"), pi("iglca_3456", "official", s, ds="lol_v1")
            for k in keysL:
                # available, consistent with Sec. VI; LPIPS exists only for the official decoding
                kk = f"{k}_s1" if all(f"{k}_s1" in r for r in list(A_.values()) + list(B_.values())) else k
                d = [B_[i][kk] - A_[i][kk] for i in sorted(A_)]
                pooled[k].extend(d)
                per_seed[k].append(d)
                base_m[k].append(float(np.mean([A_[i][kk] for i in A_])))
                ig_m[k].append(float(np.mean([B_[i][kk] for i in B_])))
        M["BLBase"], M["BLIG"] = f"{np.mean(base_m['psnr']):.2f}", f"{np.mean(ig_m['psnr']):.2f}"
        M["BLBaseGT"], M["BLIGGT"] = f"{np.mean(base_m['gt_psnr']):.2f}", f"{np.mean(ig_m['gt_psnr']):.2f}"
        M["BLGain"], M["BLGainGT"] = sgn(float(np.mean(pooled["psnr"]))), sgn(float(np.mean(pooled["gt_psnr"])))
        M["BLGainSSIM"], M["BLGainLPIPS"] = sgn(float(np.mean(pooled["ssim"])), 3), sgn(float(np.mean(pooled["lpips"])), 3)
        M["BLP"] = fmt_p(stats.wilcoxon(pooled["psnr"]).pvalue)
        M["BLPGT"] = fmt_p(stats.wilcoxon(pooled["gt_psnr"]).pvalue)
        M["BLCIPsnr"], M["BLCIGT"] = boot_ci(per_seed["psnr"]), boot_ci(per_seed["gt_psnr"])
        sg = [float(np.mean(d)) for d in per_seed["psnr"]]
        M["BLSeedGains"] = ", ".join(sgn(x) for x in sg[:-1]) + (" and " if len(sg) > 1 else "") + sgn(sg[-1]) if len(sg) > 1 else sgn(sg[0])
    else:
        for k in ("BLBase", "BLIG", "BLBaseGT", "BLIGGT", "BLGain", "BLGainGT", "BLGainSSIM", "BLGainLPIPS", "BLP", "BLPGT",
                  "BLCIPsnr", "BLCIGT", "BLSeedGains"):
            M[k] = NA

    #      (noig128); LOLv2-Real, matched 128x128 protocol, final iteration. Same protocol as for CIDNet: pooled
    #      differences, brightness part, SSIM/LPIPS, per-seed directions, hierarchical bootstrap over seeds then images.
    def rx(variant, s):
        f = os.path.join(PI, f"lolv2_real_ablation_rf_{variant}_official_s{s}_iter_150000.json")
        return json.load(open(f, encoding="utf-8"))["images"] if os.path.exists(f) else None
    seedsX = [s for s in SEEDS if rx("ig128", s) and rx("noig128", s)]
    M["BRxN"] = str(len(seedsX))
    M["BRxNWord"] = {0: "no seeds", 1: "one seed", 2: "two seeds", 3: "three seeds"}.get(len(seedsX), f"{len(seedsX)} seeds")
    if seedsX:
        dX = {k: [] for k in ("psnr", "gt_psnr", "ssim", "lpips")}
        perX = {k: [] for k in dX}
        absX = {k: {"ig": [], "no": []} for k in ("psnr", "gt_psnr")}
        for s in seedsX:
            A_, B_ = rx("noig128", s), rx("ig128", s)
            for k in dX:
                d = [B_[i][k] - A_[i][k] for i in sorted(A_)]
                dX[k].extend(d)
                perX[k].append(d)
            for k in absX:
                absX[k]["ig"].append(float(np.mean([B_[i][k] for i in B_])))
                absX[k]["no"].append(float(np.mean([A_[i][k] for i in A_])))
        mX = {k: float(np.mean(v)) for k, v in dX.items()}
        M["BRxGain"], M["BRxGainGT"] = sgn(mX["psnr"]), sgn(mX["gt_psnr"])
        M["BRxBright"] = sgn(mX["psnr"] - mX["gt_psnr"])
        M["BRxGainSSIM"], M["BRxGainLPIPS"] = sgn(mX["ssim"], 3), sgn(mX["lpips"], 3)
        for k, nm in (("psnr", "P"), ("gt_psnr", "PGT"), ("ssim", "PSSIM"), ("lpips", "PLPIPS")):
            M[f"BRx{nm}"] = fmt_p(stats.wilcoxon(dX[k]).pvalue) if np.any(np.array(dX[k]) != 0) else NA
        M["BRxCIPsnr"], M["BRxCIGT"] = boot_ci(perX["psnr"]), boot_ci(perX["gt_psnr"])
        M["BRxCISSIM"] = boot_ci([np.asarray(d) * 1000 for d in perX["ssim"]])  # in units of 1e-3
        sgX = [float(np.mean(d)) for d in perX["psnr"]]
        sgtX = [float(np.mean(d)) for d in perX["gt_psnr"]]
        joinl = lambda v: v[0] if len(v) == 1 else ", ".join(v[:-1]) + (", and " if len(v) > 2 else " and ") + v[-1]
        M["BRxSeedGains"] = joinl([sgn(x) for x in sgX])
        M["BRxSeedGT"] = joinl([sgn(x) for x in sgtX])
        M["BRxSeedBright"] = joinl([sgn(a - b) for a, b in zip(sgX, sgtX)])
        M["BRxPsnrIG"], M["BRxPsnrNoIG"] = f"{np.mean(absX['psnr']['ig']):.2f}", f"{np.mean(absX['psnr']['no']):.2f}"
        M["BRxGTIG"], M["BRxGTNoIG"] = f"{np.mean(absX['gt_psnr']['ig']):.2f}", f"{np.mean(absX['gt_psnr']['no']):.2f}"

        def cls(ci_txt):
            lo, hi = [float(x.replace("\\ensuremath{-}", "-").replace("+", "")) for x in ci_txt.strip("[]").split(", ")]
            return "positive" if lo > 0 else ("negative" if hi < 0 else "inconclusive")
        M["BRxClassPSNR"], M["BRxClassGT"] = cls(M["BRxCIPsnr"]), cls(M["BRxCIGT"])
    else:
        for k in ("BRxGain", "BRxGainGT", "BRxBright", "BRxGainSSIM", "BRxGainLPIPS", "BRxP", "BRxPGT", "BRxPSSIM",
                  "BRxPLPIPS", "BRxCIPsnr", "BRxCIGT", "BRxCISSIM", "BRxSeedGains", "BRxSeedGT", "BRxSeedBright",
                  "BRxPsnrIG", "BRxPsnrNoIG", "BRxGTIG", "BRxGTNoIG", "BRxClassPSNR", "BRxClassGT"):
            M[k] = NA

    # ---- released models of other methods, scored in the same pipeline (no output scale, as in their code)
    for tag, d in (("Rf", "retinexformer_LOLv2real"), ("Snr", "snr_LOLv2real")):
        f = os.path.join("analysis", "released_eval", d, "scores.json")
        if os.path.exists(f):
            mm = json.load(open(f))["mean"]
            M[f"B{tag}PSNR"], M[f"B{tag}GT"] = f"{mm['psnr']:.2f}", f"{mm['gt_psnr']:.2f}"
        else:
            M[f"B{tag}PSNR"], M[f"B{tag}GT"] = NA, NA
    rel_ = json.load(open("analysis/released_eval/HVI-CIDNet-LOLv2-real-bestPSNR_a0.8/scores.json"))
    M["BRelGT"] = f"{rel_['gtmean']['psnr']:.2f}"
    # Table III: released weights of three methods, same pipeline
    rel1 = "analysis/released_eval/HVI-CIDNet-LOLv2-real-bestPSNR_a1.0/scores.json"
    rows3 = []
    for name, f, scale in (("SNR-aware~\\cite{xu2022snr}", "analysis/released_eval/snr_LOLv2real/scores.json", "--"),
                           ("Retinexformer~\\cite{cai2023retinexformer}", "analysis/released_eval/retinexformer_LOLv2real/scores.json", "--")):
        if os.path.exists(f):
            mm = json.load(open(f))["mean"]
            rows3.append(f"{name} & {scale} & {mm['psnr']:.2f} & {mm['ssim']:.3f} & {mm['lpips']:.3f} & {mm['gt_psnr']:.2f}\\\\")
    rows3.append(f"CIDNet~\\cite{{yan2025hvi}} & $\\alpha{{=}}0.8$ & {rel_['plain']['psnr']:.2f} & {rel_['plain']['ssim']:.3f} & {rel_['plain']['lpips']:.3f} & {rel_['gtmean']['psnr']:.2f}\\\\")
    if os.path.exists(rel1):
        r1 = json.load(open(rel1))
        rows3.append(f"CIDNet & $\\alpha{{=}}1$ & {r1['plain']['psnr']:.2f} & {r1['plain']['ssim']:.3f} & {r1['plain']['lpips']:.3f} & {r1['gtmean']['psnr']:.2f}\\\\")
        M["BRelAoneSSIM"], M["BRelAoneLPIPS"] = f"{r1['plain']['ssim']:.3f}", f"{r1['plain']['lpips']:.3f}"
    T3 = [r"\begin{table}[!t]", r"\centering",
          r"\caption{Released LOLv2-Real weights, our pipeline. Retinexformer and SNR-aware enhancement use no output scale"
          r" in their code; CIDNet uses $\alpha=0.8$; $\alpha=1$ is best on training images.}",
          r"\label{tab:released}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
          r"\begin{tabular}{lccccc}", r"\toprule",
          r"Method & Scale & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & GT-mean$\uparrow$\\", r"\midrule",
          *rows3, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(paper, "tab_released_b.tex"), "w", encoding="utf-8").write("\n".join(T3) + "\n")

    # ---- difficulty strata (analysis/difficulty_strata.py): hardest thirds vs the opposite thirds
    fo = os.path.join("analysis", "difficulty_strata_official.json")
    if os.path.exists(fo):
        S = json.load(open(fo))["strata"]
        hard = {"input_mean": 0, "rel_noise": 2, "detail": 2}
        losses, dgts, zero = [], [], True
        for fac, h in hard.items():
            e = 2 - h
            losses.append(S[fac][e]["cid_gt"] - S[fac][h]["cid_gt"])
            dgts.append(S[fac][h]["d_gt"])
            lo, hi = S[fac][h]["d_gt_ci"]
            zero = zero and lo <= 0 <= hi
        M["BDiffLossLo"], M["BDiffLossHi"] = f"{min(losses):.1f}", f"{max(losses):.1f}"
        M["BDiffDGTLo"], M["BDiffDGTHi"] = sgn(min(dgts)), sgn(max(dgts))
        M["BDiffCIText"] = "and all three intervals include zero" if zero else "although not every interval includes zero"
    fp = os.path.join("analysis", "difficulty_strata_percep.json")
    if os.path.exists(fp):
        S = json.load(open(fp))["strata"]["input_mean"]
        M["BDiffPDark"], M["BDiffPBright"] = sgn(S[0]["d_gt"]), sgn(S[2]["d_gt"])
        M["BDiffPDarkCI"] = f"[{sgn(S[0]['d_gt_ci'][0])}, {sgn(S[0]['d_gt_ci'][1])}]"
        # per seed, darkest and brightest thirds of the input brightness (perceptual-only loss)
        F_ = json.load(open(os.path.join("analysis", "test_difficulty.json")))
        st = sorted(F_)
        qv = np.quantile([F_[k]["input_mean"] for k in st], [1 / 3, 2 / 3])
        dark = [k for k in st if F_[k]["input_mean"] <= qv[0]]
        brt = [k for k in st if F_[k]["input_mean"] > qv[1]]
        dd, db = [], []
        for s in SEEDS:
            A_, B_ = pi("lca_none", "percep", s), pi("iglca_3456", "percep", s)
            if A_ and B_:
                dd.append(float(np.mean([B_[k]["gt_psnr"] - A_[k]["gt_psnr"] for k in dark])))
                db.append(float(np.mean([B_[k]["gt_psnr"] - A_[k]["gt_psnr"] for k in brt])))
        lstp = lambda v: v[0] if len(v) == 1 else ", ".join(v[:-1]) + (", and " if len(v) > 2 else " and ") + v[-1]
        M["BDiffPDarkSeeds"] = lstp([sgn(x) for x in dd]) if dd else NA
        M["BDiffPBrightSeeds"] = lstp([sgn(x) for x in db]) if db else NA

    # ---- output scale alpha
    A = json.load(open(os.path.join("analysis", "alpha_sweep.json")))
    al = np.array(next(iter(A.values()))["alphas"])
    i08, i10 = int(np.argmin(np.abs(al - 0.8))), int(np.argmin(np.abs(al - 1.0)))
    drops = [v["test"][i08] - v["test"][i10] for v in A.values()]
    M["BAlphaDropLo"], M["BAlphaDropHi"] = f"{min(drops):.1f}", f"{max(drops):.1f}"
    tb = [v["best_alpha_test"] for v in A.values()]
    M["BAlphaTestLo"], M["BAlphaTestHi"] = f"{min(tb):.2f}", f"{max(tb):.2f}"
    trb = sorted({v["best_alpha_train"] for v in A.values()})
    M["BAlphaTrain"] = "/".join(f"{x:.2f}" for x in trb)
    r = A.get("released_bestPSNR")
    M["BRelAone"] = f"{r['test'][i10]:.2f}" if r else NA
    M["BRelAeight"] = f"{r['test'][i08]:.2f}" if r else NA
    M["BAlphaN"] = str(len(A))

    # ---- dataset brightness
    D = dataset_brightness()
    gtr, gte = np.array(D["Train"]["gt"]), np.array(D["Test"]["gt"])
    ltr, lte = np.array(D["Train"]["low"]), np.array(D["Test"]["low"])
    M["BGTTrain"], M["BGTTest"] = f"{gtr.mean():.1f}", f"{gte.mean():.1f}"
    M["BLowTrain"], M["BLowTest"] = f"{ltr.mean():.1f}", f"{lte.mean():.1f}"
    M["BGTDarker"] = f"{100 * (1 - gte.mean() / gtr.mean()):.0f}"
    M["BGTShiftP"] = fmt_p(stats.mannwhitneyu(gtr, gte).pvalue)
    M["BLowShiftP"] = fmt_p(stats.mannwhitneyu(ltr, lte).pvalue)
    rtr, rte = np.log(gtr / ltr), np.log(gte / lte)
    M["BRatioTrain"], M["BRatioTest"] = f"{np.exp(rtr.mean()):.2f}", f"{np.exp(rte.mean()):.2f}"
    M["BRatioP"] = fmt_p(stats.mannwhitneyu(rtr, rte).pvalue)
    V1 = dataset_brightness_v1()  # LOLv1: same capture protocol, no train/test shift expected
    v1tr, v1te = np.array(V1["Train"]["gt"]), np.array(V1["Test"]["gt"])
    M["BVoneGTTrain"], M["BVoneGTTest"] = f"{v1tr.mean():.1f}", f"{v1te.mean():.1f}"
    M["BVoneGTP"] = fmt_p(stats.mannwhitneyu(v1tr, v1te).pvalue)
    S1 = dataset_saturation_v1()  # LOLv1 test targets are more saturated; the official decoding scales saturation by 1.3
    s1tr, s1te = np.array(S1["Train"]), np.array(S1["Test"])
    M["BVoneSatTrain"], M["BVoneSatTest"] = f"{s1tr.mean():.1f}", f"{s1te.mean():.1f}"
    M["BVoneSatPct"] = f"{100 * (s1te.mean() / s1tr.mean() - 1):.0f}"
    M["BVoneSatP"] = fmt_p(stats.mannwhitneyu(s1tr, s1te).pvalue)
    fs = os.path.join("analysis", "sat_sweep_lolv1.json")  # analysis/sat_sweep_lolv1.py on the released LOLv1 weights
    if os.path.exists(fs):
        SW = {k: v for k, v in json.load(open(fs)).items() if k.startswith("released")}
        bt = sorted({v["test"]["best_s_psnr"] for v in SW.values()} | {v["test"]["best_s_gt"] for v in SW.values()})
        M["BVoneSatBestTest"] = f"{bt[0]:.2f}--{bt[-1]:.2f}" if len(bt) > 1 else f"{bt[0]:.2f}"
        M["BVoneSatBestTrain"] = "/".join(sorted({f"{v['train']['best_s_psnr']:.1f}" for v in SW.values()}))
        gg = [v["test"]["gt_s13"] - v["test"]["gt_s1"] for v in SW.values()]
        tl = [v["train"]["gt_s1"] - v["train"]["gt_s13"] for v in SW.values()]  # GT-mean, like the test-side gain
        M["BVoneSatGTGainLo"], M["BVoneSatGTGainHi"] = f"{min(gg):.1f}", f"{max(gg):.1f}"
        M["BVoneSatTrainLossLo"], M["BVoneSatTrainLossHi"] = f"{min(tl):.1f}", f"{max(tl):.1f}"
    else:
        for k in ("BVoneSatBestTest", "BVoneSatBestTrain", "BVoneSatGTGainLo", "BVoneSatGTGainHi", "BVoneSatTrainLossLo",
                  "BVoneSatTrainLossHi"):
            M[k] = NA

    rows = []
    for f in glob.glob(os.path.join(PI, "lolv2_real_ablation_*_percep_s3407_epoch_1000.json")):
        d = json.load(open(f, encoding="utf-8"))["images"].values()
        rows.append((np.mean([x["psnr"] for x in d]), np.mean([x["gt_psnr"] for x in d]),
                     np.mean([abs(np.log(x["bright"])) for x in d])))
    if rows:
        R = np.array(rows)
        M["BPlaceN"] = str(len(rows))
        M["BPlaceRange"] = f"{R[:, 0].max() - R[:, 0].min():.2f}"
        M["BPlaceRangeGT"] = f"{R[:, 1].max() - R[:, 1].min():.2f}"
        M["BPlaceRho"] = sgn(stats.spearmanr(R[:, 0], R[:, 2])[0])

    # ---- gates (official IG runs)
    ch, px, hv6, one, static, one_gt = [], [], [], [], [], []
    for s in O["seeds"]:
        t = f"lolv2_real_ablation_iglca_3456_official_s{s}"
        fs, fi = os.path.join("analysis", f"gate_stats_{t}.json"), os.path.join("analysis", f"gate_intervene_{t}.json")
        if os.path.exists(fs):
            G = json.load(open(fs))
            for k, v in G.items():
                if k.startswith("_") or k == "I_LCA5":
                    continue
                ch.append(v["ch_std"])
                px.append(v["px_std"])
            hv6.append(G["HV_LCA6"]["corr"])
        if os.path.exists(fi):
            I = json.load(open(fi))
            one.append(I["one"]["psnr"] - I["learned"]["psnr"])
        # the test images; the old test-set average of gate_intervene.py is no longer used in the paper
        # channel means, inference only); BGateStaticList / BGateOneList are GT-mean PSNR changes from the learned gate
        fg = os.path.join("analysis", f"gate_static_train_gt_{t}.json")
        if os.path.exists(fg):
            J = json.load(open(fg))
            static.append(J["static_train"]["gt_psnr"] - J["learned"]["gt_psnr"])
            one_gt.append(J["one"]["gt_psnr"] - J["learned"]["gt_psnr"])

    def lst2(v):
        v = [sgn(x) for x in v]
        return NA if not v else (v[0] if len(v) == 1 else ", ".join(v[:-1]) + (", and " if len(v) > 2 else " and ") + v[-1])
    M["BGateChLo"], M["BGateChHi"] = (f"{min(ch):.2f}", f"{max(ch):.2f}") if ch else (NA, NA)
    M["BGatePxLo"], M["BGatePxHi"] = (f"{min(px):.2f}", f"{max(px):.2f}") if px else (NA, NA)
    M["BGateChMean"] = f"{np.mean(ch):.2f}" if ch else NA
    M["BGatePxMean"] = f"{np.mean(px):.2f}" if px else NA
    M["BGateHVSix"] = sgn(float(np.mean(hv6))) if hv6 else NA
    M["BGateHVSixList"] = lst2(hv6)
    M["BGateOne"] = sgn(float(np.mean(one))) if one else NA
    M["BGateOneList"] = lst2(one)
    M["BGateStaticList"] = lst2(static) if len(static) == len(O["seeds"]) else NA
    M["BGateOneGTList"] = lst2(one_gt) if len(one_gt) == len(O["seeds"]) else NA
    M["BGateStaticMaxAbs"] = f"{max(abs(x) for x in static):.2f}" if static and len(static) == len(O["seeds"]) else NA
    pch, ppx = [], []
    for fs in ("gate_stats_s3407_e1000.json", "gate_stats_lolv2_real_ablation_iglca_3456_percep_s42.json",
               "gate_stats_lolv2_real_ablation_iglca_3456_percep_s1234.json"):
        fs = os.path.join("analysis", fs)
        if os.path.exists(fs):
            for k, v in json.load(open(fs)).items():
                if not k.startswith("_") and k != "I_LCA5":
                    pch.append(v["ch_std"])
                    ppx.append(v["px_std"])
    M["BGatePChMean"] = f"{np.mean(pch):.2f}" if len(pch) == 21 else NA  # 7 trained gates x 3 seeds
    M["BGatePPxMean"] = f"{np.mean(ppx):.2f}" if len(ppx) == 21 else NA

    # ---- LN representation probe (analysis/ln_probe.py; fit on every 5th training input, evaluated on the test images):
    #      is absolute intensity still linearly decodable from LN(F_I), the gate's input? IG-LCA models, LCA3-LCA6
    fp = os.path.join("analysis", "ln_probe.json")
    if os.path.exists(fp):
        LP = json.load(open(fp))
        ig = [v for k, v in LP.items() if k.startswith("iglca_3456")]
        get = lambda name, key: [lv[name][key] for r in ig for lv in r.values()]
        pre, post, rem = get("pre", "r2"), get("post", "r2"), get("removed", "r2")
        postw = get("post", "r2_within")
        M["BProbePreLo"], M["BProbePreHi"] = f"{min(pre):.2f}", f"{max(pre):.2f}"
        M["BProbePostLo"], M["BProbePostHi"] = f"{min(post):.2f}", f"{max(post):.2f}"
        M["BProbePostWithinLo"], M["BProbePostWithinHi"] = f"{min(postw):.2f}", f"{max(postw):.2f}"
        M["BProbeRemHi"] = f"{max(rem):.2f}"
        M["BProbeDiffMax"] = f"{max(abs(b - c) for b, c in zip(post, pre)):.2f}"
        M["BProbeFitN"] = LP["meta"]["fit"].split("(")[1].split()[0]
    else:
        for k in ("BProbePreLo", "BProbePreHi", "BProbePostLo", "BProbePostHi", "BProbePostWithinLo", "BProbePostWithinHi",
                  "BProbeRemHi", "BProbeDiffMax", "BProbeFitN"):
            M[k] = NA

    # ---- perceptual-only loss, seed 42: do gates without illumination input (ReZero scalar, LayerScale) also move PSNR
    #      through brightness? Each against CIDNet of the same loss and seed.
    for model, tag in (("iglca_3456_scalar", "Scalar"), ("iglca_3456_chan", "Chan")):
        A_, B_ = pi("lca_none", "percep", "42"), pi(model, "percep", "42")
        if not (A_ and B_):
            for k in ("Gain", "GainGT", "P", "PGT", "Berr", "Bright"):
                M[f"BP{tag}{k}"] = NA
            continue
        dps = [B_[i]["psnr"] - A_[i]["psnr"] for i in A_]
        dgt = [B_[i]["gt_psnr"] - A_[i]["gt_psnr"] for i in A_]
        M[f"BP{tag}Gain"], M[f"BP{tag}GainGT"] = sgn(float(np.mean(dps))), sgn(float(np.mean(dgt)))
        M[f"BP{tag}P"], M[f"BP{tag}PGT"] = fmt_p(stats.wilcoxon(dps).pvalue), fmt_p(stats.wilcoxon(dgt).pvalue)
        M[f"BP{tag}Berr"] = f"{np.mean([abs(np.log(B_[i]['bright'])) for i in B_]):.3f}"
        M[f"BP{tag}Bright"] = sgn(float(np.mean(dps)) - float(np.mean(dgt)))  # brightness part of the PSNR change
    A42 = pi("lca_none", "percep", "42")
    M["BPBerrBaseFortyTwo"] = f"{np.mean([abs(np.log(A42[i]['bright'])) for i in A42]):.3f}" if A42 else NA
    B42 = pi("iglca_3456", "percep", "42")
    M["BPGTFortyTwo"] = sgn(float(np.mean([B42[i]["gt_psnr"] - A42[i]["gt_psnr"] for i in A42]))) if A42 and B42 else NA
    M["BPGainFortyTwo"] = sgn(float(np.mean([B42[i]["psnr"] - A42[i]["psnr"] for i in A42]))) if A42 and B42 else NA
    # IG-LCA at seed 42 under the official loss, the reference for the seed-42 gate designs of Table II
    A_, B_ = pi("lca_none", "official", "42"), pi("iglca_3456", "official", "42")
    M["BOGTFortyTwo"] = sgn(float(np.mean([B_[i]["gt_psnr"] - A_[i]["gt_psnr"] for i in A_]))) if A_ and B_ else NA

    # ---- gate designs under the official loss (Table II): every variant against CIDNet over the seeds both have
    # LayerScale, ReZero, SE); one-seed rows carry an asterisk (exploratory diagnostic, caption).
    VARS = [("iglca_3456", "IG-LCA", r"$\mathrm{LN}(F_{\mathrm I})$", 48950),
            ("iglca_3456_iig", "Intensity", r"$\log L_{\downarrow}$", 25172),
            ("iglca_3456_iigf", "Feat.+int.", "both", 49148),
            ("iglca_3456_iige", "Int.+global", r"$\log L_{\downarrow},\mu,\sigma$", 25568),
            ("iglca_3456_chan", "LayerScale", "--", 792),
            ("iglca_3456_scalar", "ReZero", "--", 8),
            ("iglca_3456_se", "SE", "pooled", 48950)]
    rows_g = []
    newrows = []
    TAGS_G = {"iglca_3456_iig": "Iig", "iglca_3456_iigf": "Iigf", "iglca_3456_iige": "Iige",
              "iglca_3456_se": "Se", "iglca_3456_scalar": "Rezero", "iglca_3456_chan": "Lscale"}
    for model, name, inp, npar in VARS:
        seeds = [s for s in SEEDS if pi(model, "official", s) and pi("lca_none", "official", s)]
        if not seeds:
            miss = r"\textcolor{red}{?}"  # run still training: a red mark instead of a number
            rows_g.append(f"{name} & {inp} & +{npar:,} & {miss} & {miss} & {miss} & {miss}\\\\")
            newrows.append((model, None))
            tag = TAGS_G.get(model)
            if tag:
                for k in ("N", "NWord", "Gain", "P", "GainAone", "PAone", "GainGT", "PGT", "GainSSIM", "PSSIM",
                          "GainLPIPS", "PLPIPS", "Berr", "SeedAone", "SeedGT", "CIGT", "SeedAoneNoP"):
                    M[f"B{tag}{k}"] = NA
            continue
        vals = {k: [] for k in ("psnr", "psnr_a1", "gt_psnr")}
        d = {k: [] for k in vals}
        for s in seeds:
            A_, B_ = pi("lca_none", "official", s), pi(model, "official", s)
            for k in vals:
                vals[k].append(np.mean([B_[i][k] for i in B_]))
                d[k].extend([B_[i][k] - A_[i][k] for i in A_])
        cells = [f"{np.mean(vals[k]):.2f}" for k in ("psnr", "gt_psnr")]
        dgt = float(np.mean(d["gt_psnr"]))
        nseed = f"{len(seeds)}" + (r"$^{*}$" if len(seeds) == 1 else "")
        per_gt = []
        for s in seeds:
            A_, B_ = pi("lca_none", "official", s), pi(model, "official", s)
            per_gt.append([B_[i]["gt_psnr"] - A_[i]["gt_psnr"] for i in sorted(A_)])
        ci_txt = f" {{\\scriptsize {boot_ci(per_gt)}}}" if len(seeds) > 1 else ""
        rows_g.append(f"{name} & {inp} & +{npar:,} & {nseed} & " + " & ".join(cells) + f" & {sgn(dgt)}{ci_txt}\\\\")
        newrows.append((model, (vals, dgt, len(seeds))))
        tag = TAGS_G.get(model)
        if tag:
            M[f"B{tag}N"] = str(len(seeds))
            M[f"B{tag}NWord"] = {1: "one seed", 2: "two seeds", 3: "three seeds"}.get(len(seeds), f"{len(seeds)} seeds")
            for k, nm in (("psnr", ""), ("psnr_a1", "Aone"), ("gt_psnr", "GT")):
                M[f"B{tag}Gain{nm}"] = sgn(float(np.mean(d[k])))
                M[f"B{tag}P{nm}"] = fmt_p(stats.wilcoxon(d[k]).pvalue)
            for k, nm in (("ssim", "SSIM"), ("lpips", "LPIPS")):
                dd = [pi(model, "official", s)[i][k] - pi("lca_none", "official", s)[i][k] for s in seeds for i in pi("lca_none", "official", s)]
                M[f"B{tag}Gain{nm}"] = sgn(float(np.mean(dd)), 3)
                M[f"B{tag}P{nm}"] = fmt_p(stats.wilcoxon(dd).pvalue)
            be = [abs(np.log(pi(model, "official", s)[i]["bright"])) for s in seeds for i in pi(model, "official", s)]
            M[f"B{tag}Berr"] = f"{np.mean(be):.3f}"
            # per seed (seed order of SEEDS): alpha-free PSNR difference with its own p, and the GT-mean difference
            seed_a1, seed_gt, per_seed_gt, seed_a1_v = [], [], [], []
            for s in seeds:
                A_, B_ = pi("lca_none", "official", s), pi(model, "official", s)
                da = [B_[i]["psnr_a1"] - A_[i]["psnr_a1"] for i in A_]
                seed_a1_v.append(float(np.mean(da)))
                seed_a1.append(f"{sgn(float(np.mean(da)))}~dB (\\ensuremath{{p={fmt_p(stats.wilcoxon(da).pvalue)}}})")
                per_seed_gt.append([B_[i]["gt_psnr"] - A_[i]["gt_psnr"] for i in sorted(A_)])
                seed_gt.append(float(np.mean(per_seed_gt[-1])))
            M[f"B{tag}CIGT"] = boot_ci(per_seed_gt) if len(seeds) > 1 else NA  # seeds, then images
            M[f"B{tag}SeedAone"] = seed_a1[0] if len(seed_a1) == 1 else (
                ", ".join(seed_a1[:-1]) + (", and " if len(seed_a1) > 2 else " and ") + seed_a1[-1])
            M[f"B{tag}SeedGT"] = lst2(seed_gt)
            M[f"B{tag}SeedAoneNoP"] = lst2(seed_a1_v)
    base3 = [s for s in SEEDS if pi("lca_none", "official", s)]
    bv = {k: [np.mean([x[k] for x in pi("lca_none", "official", s).values()]) for s in base3] for k in ("psnr", "psnr_a1", "gt_psnr")}
    best = []
    for s in base3:
        logs = sorted(glob.glob(os.path.join("results", "training", f"lolv2_real_ablation_lca_none_official_s{s}",
                                             "best_metrics_*.log")))
        m_ = re.search(r"Avg\.PSNR:\s*([\d.]+)", open(logs[-1], encoding="utf-8", errors="ignore").read()) if logs else None
        if m_:
            best.append(float(m_.group(1)))
    ok = len(best) == len(base3) and best
    M["BOBestEpochMean"] = f"{np.mean(best):.2f}" if ok else NA
    M["BOBestEpochMax"] = f"{max(best):.2f}" if ok else NA
    TG = [r"\begin{table}[!t]", r"\centering",
          r"\caption{Gate designs at LCA3--LCA6, official loss. Params: added to CIDNet's 1,975,569."
          r" $\Delta$GT: GT-mean PSNR minus CIDNet of the same seeds; brackets: hierarchical 95\% interval."
          r" $^{*}$One seed: exploratory, not primary evidence.}",
          r"\label{tab:gate_ablation}", r"\scriptsize", r"\setlength{\tabcolsep}{1.6pt}",
          r"\begin{tabular}{llrcccc}", r"\toprule",
          r"Gate & Input & Params & Seeds & PSNR$_{0.8}$ & GT-mean & $\Delta$GT\\", r"\midrule",
          f"CIDNet & no gate & 0 & {len(base3)} & " + " & ".join(f"{np.mean(bv[k]):.2f}" for k in ("psnr", "gt_psnr")) + r" & --\\",
          *rows_g, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(paper, "tab_gates_b_oldstyle.tex"), "w", encoding="utf-8").write("\n".join(TG) + "\n")
    LAYOUT = {
        "iglca_3456_chan": ("LayerScale", "touvron2021going", "--", "u"),
        "iglca_3456_scalar": ("ReZero", "bachlechner2021rezero", "--", "u"),
        "iglca_3456_se": ("SE", "hu2018squeeze", r"Pooled $F$", "u"),
        "iglca_3456_iig": ("Intensity Gate", None, r"$\log L_{\downarrow}$", "g"),
        "iglca_3456_iigf": ("Feat.+Int. Gate", None, r"$\mathrm{LN}(F_{\mathrm I}), \log L_{\downarrow}$", "g"),
        "iglca_3456_iige": ("Int.+Global Gate", None, r"$\log L_{\downarrow}, \mu, \sigma$", "g"),
        "iglca_3456": ("IG-LCA (Ours)", None, r"$\mathrm{LN}(F_{\mathrm I})$", "g")}
    NPAR = {m: n for m, _, _, n in VARS}
    DATA = dict(newrows)

    def kpar(n):
        return f"+{n / 1000:.1f}K" if n >= 100 else f"+{n}"

    def pm(v):  # mean and population std over seeds, as in Table I; a single seed has no spread
        return f"{np.mean(v):.2f}$\\pm${np.std(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"

    def dpar(x):
        if round(x, 2) > 0:
            return f"(\\pos{{{x:.2f}}})"
        return f"(\\nega{{{abs(x):.2f}}})" if round(x, 2) < 0 else "(0.00)"

    def row(m):
        name, cite, inp, _ = LAYOUT[m]
        dat = DATA.get(m)
        # planned for seed 42 only
        dag = r"$^{*}$" if dat is None or dat[2] == 1 else ""
        head = f"{name}{dag}" + (f"~\\cite{{{cite}}}" if cite else "")
        if dat is None:  # still training: the red [TBD] mark of the paper, never a number
            return f"{head} & {inp} & {kpar(NPAR[m])} & \\TBD & \\TBD \\\\"
        vals, dgt, _ = dat
        return f"{head} & {inp} & {kpar(NPAR[m])} & {pm(vals['psnr'])} & {pm(vals['gt_psnr'])} {dpar(dgt)} \\\\"

    TN = [r"\begin{table}[t]", r"\centering",
          r"\caption{Gate designs at LCA3--LCA6, official loss. Mean$\pm$population std over three"
          r" seeds; $^{*}$one seed (exploratory). Parentheses: GT-mean PSNR difference from same-seed CIDNet.}",
          r"\label{tab:gate_ablation}", r"\resizebox{\columnwidth}{!}{", r"\begin{tabular}{l l r c l}", r"\toprule",
          r"Gate Mechanism & Input Feature & Params ($\Delta$) & PSNR$_{0.8}\uparrow$ & GT-mean$\uparrow$ \\",
          r"\midrule", f"CIDNet (Baseline) & None & 0 & {pm(bv['psnr'])} & {pm(bv['gt_psnr'])} \\\\", r"\midrule",
          r"\multicolumn{5}{l}{\textit{Unguided / Spatial-agnostic}} \\",
          *[row(m) for m in LAYOUT if LAYOUT[m][3] == "u"], r"\midrule",
          r"\multicolumn{5}{l}{\textit{Illumination-Guided}} \\",
          *[row(m) for m in LAYOUT if LAYOUT[m][3] == "g"],
          r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    # layout is kept for reference only
    open(os.path.join(paper, "tab_gates_b_layout1009.tex"), "w", encoding="utf-8").write("\n".join(TN) + "\n")

    # same loss and seed; signed values, positive red (\pos) and negative blue (\nega), no Delta symbol
    def sdiff(x, nd):
        v = round(x, nd)
        if v > 0:
            return f"\\pos{{{v:.{nd}f}}}"
        if v < 0:
            return f"\\nega{{{abs(v):.{nd}f}}}"
        return f"{0:.{nd}f}"

    def paired4(base_tag, model_tag, recipe, seeds):
        """mean over seeds of the per-image paired differences; None if a run is missing"""
        out = {k: [] for k in ("psnr", "gt_psnr", "ssim", "lpips")}
        for s in seeds:
            A_, B_ = pi(base_tag, recipe, s), pi(model_tag, recipe, s)
            if not (A_ and B_):
                return None
            for k in out:
                out[k].append(float(np.mean([B_[i][k] - A_[i][k] for i in A_])))
        return {k: float(np.mean(v)) for k, v in out.items()}

    def cells4(d):
        return " & ".join([sdiff(d["psnr"], 2), sdiff(d["gt_psnr"], 2), sdiff(d["ssim"], 3), sdiff(d["lpips"], 3)])

    # Table A: architecture ablation (placement and branch), perceptual-only loss, seed 3407 (one run each)
    def active(cfg):  # gates whose output reaches the network: the official forward discards the I_LCA5 output
        return sum(1 + (k != "5") for k in cfg)
    A0 = pi("lca_none", "percep", "3407")
    # tall and pushed the conclusion to page 6. The one-column layout is still written to tab_ablation_arch_col.tex.
    cfgs_t = [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), ("6", "6"), ("12", "1,2"), ("34", "3,4"),
              ("345", "3--5"), ("456", "4--6"), ("1234", "1--4"), ("2345", "2--5"), ("3456", r"\textbf{3--6}"),
              ("123456", "1--6")]
    dT = {c: paired4("lca_none", f"iglca_{c}", "percep", ["3407"]) for c, _ in cfgs_t}
    dT["hv"] = paired4("lca_none", "iglca_3456_hv_only", "percep", ["3407"])
    dT["i"] = paired4("lca_none", "iglca_3456_i_only", "percep", ["3407"])
    colsT = [c for c, _ in cfgs_t] + ["hv", "i"]
    gatesT = {**{c: active(c) for c, _ in cfgs_t}, "hv": 4, "i": 3}
    TT = [r"\begin{table*}[!t]", r"\centering",
          r"\caption{Architecture ablation, LOLv2-Real, perceptual-only loss, seed 3407, final checkpoint (one run each,"
          r" exploratory): differences from CIDNet (absolute values in its column). Columns: gated LCA pairs; Gates:"
          r" gates whose output reaches the network. LCA3--LCA6 (bold) was selected by PSNR. LPIPS: lower is better.}",
          r"\label{tab:arch}", r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
          r"\begin{tabular}{l c *{6}{c} *{8}{c} cc}", r"\toprule",
          r" & & \multicolumn{6}{c}{Single LCA pair} & \multicolumn{8}{c}{Combined LCA pairs} & \multicolumn{2}{c}{LCA3--6, one branch}\\",
          r"\cmidrule(lr){3-8}\cmidrule(lr){9-16}\cmidrule(lr){17-18}",
          r"IG-LCA at & CIDNet & " + " & ".join(lab for _, lab in cfgs_t) + r" & HV & I\\", r"\midrule",
          r"Gates & 0 & " + " & ".join(str(gatesT[c]) for c in colsT) + r"\\"]
    for k, nd, lab in (("psnr", 2, "PSNR"), ("gt_psnr", 2, "GT-mean"), ("ssim", 3, "SSIM"), ("lpips", 3, "LPIPS")):
        base = f"{np.mean([v[k] for v in A0.values()]):.{nd}f}" if A0 else r"\TBD"
        TT.append(f"{lab} & {base} & " + " & ".join(sdiff(dT[c][k], nd) if dT[c] else r"\TBD" for c in colsT) + r"\\")
    TT += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(paper, "tab_ablation_arch.tex"), "w", encoding="utf-8").write("\n".join(TT) + "\n")
    TA = [r"\begin{table}[t]", r"\centering",
          r"\caption{Architecture ablation, LOLv2-Real, perceptual-only loss, seed 3407, final checkpoint (one run"
          r" each, exploratory): differences from CIDNet of the same seed (first row, absolute values). Gates: gates"
          r" whose output reaches the network. LCA3--LCA6 was selected from this table by PSNR. LPIPS: lower is"
          r" better.}",
          r"\label{tab:arch}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
          r"\begin{tabular}{lccccc}", r"\toprule",
          r"IG-LCA at & Gates & PSNR & GT-mean & SSIM & LPIPS\\", r"\midrule"]
    if A0:
        TA.append("none (CIDNet) & 0 & " + " & ".join(
            f"{np.mean([v[k] for v in A0.values()]):.{nd}f}" for k, nd in
            (("psnr", 2), ("gt_psnr", 2), ("ssim", 3), ("lpips", 3))) + r"\\")
    TA.append(r"\midrule")
    groups = ((("1", "2", "3", "4", "5", "6"), "single"),
              (("12", "34", "345", "456", "1234", "2345", "3456", "123456"), "combined"))
    for cfgs, _ in groups:
        for cfg in cfgs:
            d = paired4("lca_none", f"iglca_{cfg}", "percep", ["3407"])
            name = "LCA" + (cfg if len(cfg) == 1 else f"{','.join(cfg)}")
            if cfg == "123456":
                name = "LCA1--6 (all)"
            elif cfg == "3456":
                name = r"LCA3--6 (both branches)"
            TA.append(f"{name} & {active(cfg)} & " + (cells4(d) if d else r"\TBD & \TBD & \TBD & \TBD") + r"\\")
        TA.append(r"\midrule")
    for tag, name, n in (("iglca_3456_hv_only", "LCA3--6, HV branch only", 4),
                         ("iglca_3456_i_only", "LCA3--6, I branch only", 3)):
        d = paired4("lca_none", tag, "percep", ["3407"])
        TA.append(f"{name} & {n} & " + (cells4(d) if d else r"\TBD & \TBD & \TBD & \TBD") + r"\\")
    TA += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(paper, "tab_ablation_arch_col.tex"), "w", encoding="utf-8").write("\n".join(TA) + "\n")

    # Table II (Table B): gate designs, official loss, paired with CIDNet of the same seed(s)
    GB = [("iglca_3456", "IG-LCA", r"$\mathrm{LN}(F_{\mathrm I})$", 48950, ["3407", "42", "1234"], "g"),
          ("iglca_3456_iig", "Intensity", r"$\log L_{\downarrow}$", 25172, ["3407", "42", "1234"], "g"),
          ("iglca_3456_iigf", "Feat.+int.$^{*}$", r"$\mathrm{LN}(F_{\mathrm I}),\log L_{\downarrow}$", 49148, ["3407"], "g"),
          ("iglca_3456_iige", "Int.+global$^{*}$", r"$\log L_{\downarrow},\mu,\sigma$", 25568, ["3407"], "g"),
          ("iglca_3456_chan", r"LayerScale$^{*}$~\cite{touvron2021going}", "--", 792, ["42"], "u"),
          ("iglca_3456_scalar", r"ReZero$^{*}$~\cite{bachlechner2021rezero}", "--", 8, ["42"], "u"),
          ("iglca_3456_se", r"SE$^{*}$~\cite{hu2018squeeze}", "pooled $F$", 48950, ["42"], "u"),
          ("iglca_3456", r"IG-LCA, seed 42$^{*}$", r"$\mathrm{LN}(F_{\mathrm I})$", 48950, ["42"], "r")]
    TB = [r"\begin{table}[t]", r"\centering",
          # keep the asterisk
          r"\caption{Gate-design ablation at LCA3--LCA6, LOLv2-Real, official loss, final checkpoint: differences"
          r" from CIDNet of the same seeds (first row: absolute values), mean over three seeds; $^{*}$one seed"
          r" (exploratory); the last row is the seed-42 reference for the generic gates. LPIPS: lower is better.}",
          r"\label{tab:gate_ablation}", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
          r"\begin{tabular}{lcrcccc}", r"\toprule",
          r"Gate & Input & Params & PSNR & GT-mean & SSIM & LPIPS\\", r"\midrule",
          "CIDNet & -- & 0 & " + " & ".join(
              f"{np.mean([np.mean([v[k] for v in pi('lca_none', 'official', s).values()]) for s in base3]):.{nd}f}"
              for k, nd in (("psnr", 2), ("gt_psnr", 2), ("ssim", 3), ("lpips", 3))) + r"\\",
          r"\midrule", r"\multicolumn{7}{l}{\textit{Illumination-conditioned}}\\"]
    last = None
    for model, name, inp, npar, seeds, grp in GB:
        if grp != last and last is not None:
            TB.append(r"\midrule")
            if grp == "u":
                TB.append(r"\multicolumn{7}{l}{\textit{Without illumination input}}\\")
        last = grp
        d = paired4("lca_none", model, "official", seeds)
        TB.append(f"{name} & {inp} & {kpar(npar)} & "
                  + (cells4(d) if d else r"\TBD & \TBD & \TBD & \TBD") + r"\\")
    TB += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(paper, "tab_gates_b.tex"), "w", encoding="utf-8").write("\n".join(TB) + "\n")

    # ---- write macros
    lines = ["% generated by analysis/make_numbers_b.py -- do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    open(os.path.join(paper, "numbers_b.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---- main table
    rel = json.load(open("analysis/released_eval/HVI-CIDNet-LOLv2-real-bestPSNR_a0.8/scores.json"))

    def rowm(R, who):
        d = R[who]
        return " & ".join([ms(d["psnr"]), ms(d["psnr_a1"]), ms(d["gt_psnr"]), ms(d["ssim"], 3), ms(d["lpips"], 3), ms(d["berr"], 3)])

    def seed_diffs(rec, key, model="iglca_3456"):
        out = []
        for s in SEEDS:
            A_, B_ = pi("lca_none", rec, s), pi(model, rec, s)
            if not (A_ and B_):
                continue
            if key == "berr":
                out.append([abs(np.log(B_[i]["bright"])) - abs(np.log(A_[i]["bright"])) for i in sorted(A_)])
            elif all(key in A_[i] and key in B_[i] for i in A_):
                out.append([B_[i][key] - A_[i][key] for i in sorted(A_)])
        return out

    def ci_nd(per_seed, nd=2, B=20000):
        rng = np.random.default_rng(0)
        D = [np.asarray(d) for d in per_seed]
        Mimg = np.stack([d[rng.integers(0, len(d), (B, len(d)))].mean(1) for d in D], axis=1)
        pick = rng.integers(0, len(D), (B, len(D)))
        lo, hi = np.percentile(np.take_along_axis(Mimg, pick, axis=1).mean(1), [2.5, 97.5])
        return f"[{sgn(float(lo), nd)}, {sgn(float(hi), nd)}]"

    def rowd(R, rec):
        cells = []
        for k, nd in (("psnr", 2), ("psnr_a1", 2), ("gt_psnr", 2), ("ssim", 3), ("lpips", 3), ("berr", 3)):
            v = R["dm"][k]
            per = seed_diffs(rec, k)
            # intervals for the three PSNR columns only: with all six the table is 63 pt wider than the page
            ci = ci_nd(per, nd) if len(per) > 1 and k in ("psnr", "psnr_a1", "gt_psnr") else ""
            cells.append(NA if v is None else f"{sdiff(v, nd)}" + (f" {{\\scriptsize {ci}}}" if ci else ""))
        return " & ".join(cells)

    rel_rows = []
    for name, f in (("SNR-aware~\\cite{xu2022snr}", "analysis/released_eval/snr_LOLv2real/scores.json"),
                    ("Retinexformer~\\cite{cai2023retinexformer}", "analysis/released_eval/retinexformer_LOLv2real/scores.json")):
        if os.path.exists(f):
            mm = json.load(open(f))["mean"]
            rel_rows.append(f"{name} & -- & {mm['psnr']:.2f} & {mm['gt_psnr']:.2f} & {mm['ssim']:.3f} & {mm['lpips']:.3f} & --")
        else:
            rel_rows.append(f"{name} & -- & {NA} & {NA} & {NA} & {NA} & --")
    rel_rows.append(f"CIDNet~\\cite{{yan2025hvi}} & {rel['plain']['psnr']:.2f} & {M['BRelAone']} & {rel['gtmean']['psnr']:.2f} & {rel['plain']['ssim']:.3f} & {rel['plain']['lpips']:.3f} & --")
    T = [r"\begin{table*}[!t]", r"\centering",
         # Sec. III) to keep the conclusion on page 4
         r"\caption{LOLv2-Real. Top: released weights, our pipeline (no output scale for SNR-aware and Retinexformer;"
         r" SSIM and LPIPS at released settings). Below: final checkpoint,"
         r" mean\,$\pm$\,population std over seeds. PSNR with the official output scale ($\alpha=0.8$) and without ($\alpha=1$, best"
         r" on training images). Difference: IG-LCA minus CIDNet (red: $+$, blue: $-$); brackets: hierarchical-bootstrap 95\% interval.}",
         r"\label{tab:main}", r"\footnotesize", r"\setlength{\tabcolsep}{2.0pt}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Loss & Model & PSNR$\uparrow$, $\alpha{=}0.8$ & PSNR$\uparrow$, $\alpha{=}1$ & GT-mean PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & Bright.\ err.$\downarrow$\\",
         r"\midrule",
         f"\\multirow{{3}}{{*}}{{\\shortstack[l]{{Released\\\\weights}}}} & {rel_rows[0]}\\\\",
         f" & {rel_rows[1]}\\\\",
         f" & {rel_rows[2]}\\\\",
         r"\midrule"]
    for name, R in (("Perceptual only", P), ("Official", O)):
        n = len(R["seeds"])
        T.append(f"\\multirow{{3}}{{*}}{{\\shortstack[l]{{{name}\\\\({n} seed{'s' if n != 1 else ''})}}}} & CIDNet & {rowm(R, 'base')}\\\\")
        T.append(f" & IG-LCA-3456 & {rowm(R, 'ig')}\\\\")
        T.append(f" & Difference & {rowd(R, 'percep' if name == 'Perceptual only' else 'official')}\\\\")
        T.append(r"\midrule" if name != "Official" else r"\bottomrule")
    T += [r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(paper, "tab_main_b.tex"), "w", encoding="utf-8").write("\n".join(T) + "\n")
    print(json.dumps(M, indent=0, ensure_ascii=False))


if __name__ == "__main__":
    main()
