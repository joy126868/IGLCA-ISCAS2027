# -*- coding: utf-8 -*-
"""One-time setup for the Retinexformer illumination-guidance ablation (cross-architecture diagnostic).

1. Copies a clone of the official Retinexformer code (https://github.com/caiyuanhao1998/Retinexformer, given as the
   first argument; read only) to third_party/Retinexformer_iscas, without experiments/results/weights/data/figures.
   Then run third_party/patch_rf_ig.py once to add the switch that removes the illumination modulation of IG-MSA.
2. The import-only lmdb stub in third_party/rf_stubs lets the official basicsr run on Windows.
3. Builds datasets_rf/LOLv2_real/{Train,Test}/{input,target} with HARD LINKS to the LOLv2-Real images in
   LOLV2/LOL-v2/Real_captured, renamed low00001.png / normal00001.png -> 00001.png, because basicsr's paired loader
   matches input and target by file name. No pixel data is copied.
Idempotent: existing copies and links are left as they are.
Usage: python third_party/setup_rf_copy.py <path to a clone of the official Retinexformer repository>
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Retinexformer")
DST = os.path.join(HERE, "Retinexformer_iscas")
STUB_SRC = os.path.join(HERE, "rf_stubs", "lmdb.py")
STUB_DST = STUB_SRC
LOL = os.path.join(REPO, "LOLV2", "LOL-v2", "Real_captured")
DATA = os.path.join(REPO, "datasets_rf", "LOLv2_real")


def main():
    if not os.path.isdir(DST):
        top_level_skip = {"experiments", "results", "pretrained_weights", "data", "figure", ".git"}

        def ignore(d, names):  # skip these only at the top level: basicsr/data is code and must be copied
            skip = top_level_skip if os.path.abspath(d) == os.path.abspath(SRC) else set()
            return [n for n in names if n in skip or n == "__pycache__" or n.endswith(".pth")]
        shutil.copytree(SRC, DST, ignore=ignore)
        print("copied code to", DST)
    # the official basicsr/ has no __init__.py (namespace package), so a pip-installed basicsr in the env can win the
    # import; an empty __init__.py in the COPY makes it a regular package that wins when ROOT is first on sys.path
    init = os.path.join(DST, "basicsr", "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
    os.makedirs(os.path.dirname(STUB_DST), exist_ok=True)
    if not os.path.exists(STUB_DST):
        shutil.copy2(STUB_SRC, STUB_DST)
    n = 0
    for split in ("Train", "Test"):
        for kind, sub, prefix in (("input", "Low", "low"), ("target", "Normal", "normal")):
            src_dir, dst_dir = os.path.join(LOL, split, sub), os.path.join(DATA, split, kind)
            os.makedirs(dst_dir, exist_ok=True)
            for f in sorted(os.listdir(src_dir)):
                if not (f.startswith(prefix) and f.endswith(".png")):
                    continue
                dst = os.path.join(dst_dir, f[len(prefix):])
                if not os.path.exists(dst):
                    os.link(os.path.join(src_dir, f), dst)
                    n += 1
    print("hard links created:", n)
    for split in ("Train", "Test"):
        a, b = (sorted(os.listdir(os.path.join(DATA, split, k))) for k in ("input", "target"))
        assert a == b, f"{split}: input and target names differ"
        print(split, len(a), "pairs, e.g.", a[0])


if __name__ == "__main__":
    main()
