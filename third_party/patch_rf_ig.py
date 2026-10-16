import os
# -*- coding: utf-8 -*-
"""Add an `ig` switch to the COPIED Retinexformer architecture (third_party/Retinexformer_iscas only).

ig=True  (default): official IG-MSA, values multiplied by the illumination features (v = v * illu_attn)
ig=False          : the same network without that multiplication (illumination guidance removed); the illumination
                    estimator and the light-up step (img * illu_map + img) stay, and no parameter is added or removed.
Every replacement must match exactly once; the script refuses to patch twice.
"""
import re

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Retinexformer_iscas", "basicsr", "models", "archs", "RetinexFormer_arch.py")
s = open(P, encoding="utf-8").read()
assert "self.ig = ig" not in s, "already patched"

REPL = [
    # IG_MSA: switch
    ("            dim_head=64,\n            heads=8,\n    ):\n        super().__init__()\n        self.num_heads = heads",
     "            dim_head=64,\n            heads=8,\n            ig=True,\n    ):\n        super().__init__()\n        self.ig = ig  # ISCAS ablation\n        self.num_heads = heads"),
    ("        v = v * illu_attn\n",
     "        if self.ig:  # ISCAS ablation: ig=False removes the illumination guidance\n            v = v * illu_attn\n"),
    # IGAB: pass through
    ("            heads=8,\n            num_blocks=2,\n    ):",
     "            heads=8,\n            num_blocks=2,\n            ig=True,\n    ):"),
    ("                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),",
     "                IG_MSA(dim=dim, dim_head=dim_head, heads=heads, ig=ig),"),
    # Denoiser
    ("    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):",
     "    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4], ig=True):"),
    ("                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),",
     "                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim, ig=ig),"),
    ("            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])",
     "            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1], ig=ig)"),
    ("                    heads=(dim_level // 2) // dim),",
     "                    heads=(dim_level // 2) // dim, ig=ig),"),
    # single stage and top level
    ("    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):",
     "    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1], ig=True):"),
    ("level=level,num_blocks=num_blocks)",
     "level=level,num_blocks=num_blocks,ig=ig)"),
    ("    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1]):",
     "    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1], ig=True):"),
    ("n_feat=n_feat, level=2, num_blocks=num_blocks)",
     "n_feat=n_feat, level=2, num_blocks=num_blocks, ig=ig)"),
]
for old, new in REPL:
    n = s.count(old)
    assert n == 1, f"pattern found {n} times: {old!r}"
    s = s.replace(old, new)
open(P, "w", encoding="utf-8").write(s)
print("patched", P, "-", len(REPL), "replacements; ig occurrences:", len(re.findall(r"\big\b", s)))
