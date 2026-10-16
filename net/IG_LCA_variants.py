# -*- coding: utf-8 -*-
"""Gate-design variants of IG-LCA.

Every variant keeps the LCA computation of net/IG_LCA.py and changes only the gate G that scales the
cross-branch residual (x = x + G * attn):
  ig      : reference IG-LCA. G = 2*sigmoid(g(guide)), g = 1x1 conv -> PReLU -> 1x1 conv (last conv
            zero-initialised, so G = 1 at start). Guide = I-branch feature in both branch updates.
  self    : as ig, but the HV update is guided by its own HV feature (no illumination guidance).
  se      : SE-style channel gate from the branch's own feature: global average pool -> g -> 2*sigmoid.
  scalar  : ReZero-style learnable scalar per block, initialised to 1 (no input dependence).
  chan    : LayerScale-style learnable per-channel scale, initialised to 1 (no input dependence).
  sig     : standard sigmoid gate G = sigmoid(g(guide)) in [0, 1] (starts at 0.5).
  zero    : as ig, but the last bias starts at -6.9 so G = 2*sigmoid(-6.9) ~ 0.002 (gate starts closed).
  linear  : as ig, but g is a single 1x1 conv (no bottleneck).
  iig     : input-illumination gate. G = 2*sigmoid(g(log L)), L = the input intensity map
            (HVI I channel = max(R,G,B)) average-pooled to the block resolution. The ig gate reads features after
            the per-pixel LayerNorm, which removes the pixel's brightness, and learned an almost spatially
            constant per-channel scale; iig sees the actual illumination. The generated CIDNet sets ILLUM['map'].
  iigf    : as iig, but g reads [LayerNorm'd I-branch feature, log L] (features plus illumination).
  iige    : as iig, plus the scene exposure: g reads [log L, mean(log L), std(log L)], where mean and
            std are taken over the whole input image (an image-level exposure prior in the spirit of the
            prior-guided attention of HVI-CIDNet+, but two numbers instead of a vision-language embedding).
            Motivation: iig gained in bright scenes and lost in dark scenes at seed 3407, so the response to a
            pixel's brightness should depend on the exposure of the whole scene.
Usage: IGLCA_VARIANT=<name> selects the classes IG_HV_LCA_<name> / IG_I_LCA_<name> (see make_classes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.transformer_utils import LayerNorm
from net.LCA import CAB, IEL


class BottleneckGate(nn.Module):
    def __init__(self, dim, reduction=4, act="2sig", bias_init=0.0, pooled=False, bottleneck=True):
        super().__init__()
        self.act, self.pooled = act, pooled
        if bottleneck:
            hidden = max(dim // reduction, 8)
            self.body = nn.Sequential(nn.Conv2d(dim, hidden, 1, bias=True), nn.PReLU(), nn.Conv2d(hidden, dim, 1, bias=True))
        else:
            self.body = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=True))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, bias_init)

    def forward(self, guide):
        if self.pooled:
            guide = F.adaptive_avg_pool2d(guide, 1)
        z = self.body(guide)
        return 2.0 * torch.sigmoid(z) if self.act == "2sig" else torch.sigmoid(z)


class ScalarGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, guide):
        return self.alpha


class ChannelScaleGate(nn.Module):
    """LayerScale-style learnable per-channel scale, initialised to 1 (static: no input dependence)."""
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, guide):
        return self.alpha


ILLUM = {"map": None}  # input intensity map (B, 1, H, W), set by the generated CIDNet before each forward pass


class InputIllumGate(nn.Module):
    """G = 2*sigmoid(g(z)), z = log of the input intensity pooled to the feature size (and the guide if with_feat)."""
    def __init__(self, dim, reduction=4, with_feat=False, with_global=False):
        super().__init__()
        self.with_feat, self.with_global = with_feat, with_global
        hidden = max(dim // reduction, 8)
        n_in = (dim if with_feat else 0) + 1 + (2 if with_global else 0)
        self.body = nn.Sequential(nn.Conv2d(n_in, hidden, 1, bias=True), nn.PReLU(),
                                  nn.Conv2d(hidden, dim, 1, bias=True))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, guide):
        illum = ILLUM["map"]
        if illum is None:
            raise RuntimeError("ILLUM['map'] is not set: use the generated CIDNet_IGLCA_ABLATION_*_iig model")
        logl = torch.log(F.adaptive_avg_pool2d(illum, guide.shape[-2:]) + 1e-3).to(guide.dtype)
        parts = [guide, logl] if self.with_feat else [logl]
        if self.with_global:  # scene exposure: mean and std of log intensity over the whole input image
            full = torch.log(illum + 1e-3).to(guide.dtype)
            mu = full.mean(dim=(2, 3), keepdim=True)
            sd = full.std(dim=(2, 3), keepdim=True)
            parts += [mu.expand_as(logl), sd.expand_as(logl)]
        z = torch.cat(parts, dim=1)
        return 2.0 * torch.sigmoid(self.body(z))


GATES = {
    "ig": lambda d: BottleneckGate(d),
    "self": lambda d: BottleneckGate(d),
    "se": lambda d: BottleneckGate(d, pooled=True),
    "scalar": lambda d: ScalarGate(d),
    "chan": lambda d: ChannelScaleGate(d),
    "sig": lambda d: BottleneckGate(d, act="sig"),
    "zero": lambda d: BottleneckGate(d, bias_init=-6.9),
    "linear": lambda d: BottleneckGate(d, bottleneck=False),
    "iig": lambda d: InputIllumGate(d),
    "iigf": lambda d: InputIllumGate(d, with_feat=True),
    "iige": lambda d: InputIllumGate(d, with_global=True),
}


def make_classes(variant):
    make_gate = GATES[variant]
    hv_guide_self = variant in ("self", "se")  # HV update guided by its own HV feature

    class HV(nn.Module):
        def __init__(self, dim, num_heads, bias=False):
            super().__init__()
            self.norm = LayerNorm(dim)
            self.ffn = CAB(dim, num_heads, bias=bias)
            self.gdfn = IEL(dim)
            self.illum_gate = make_gate(dim)

        def forward(self, x, y):
            x_norm, y_norm = self.norm(x), self.norm(y)
            attn = self.ffn(x_norm, y_norm)
            gate = self.illum_gate(x_norm if hv_guide_self else y_norm)
            x = x + gate * attn
            return self.gdfn(self.norm(x))  # same form as HV_LCA

    class I(nn.Module):
        def __init__(self, dim, num_heads, bias=False):
            super().__init__()
            self.norm = LayerNorm(dim)
            self.ffn = CAB(dim, num_heads, bias=bias)
            self.gdfn = IEL(dim)
            self.illum_gate = make_gate(dim)

        def forward(self, x, y):
            x_norm, y_norm = self.norm(x), self.norm(y)
            attn = self.ffn(x_norm, y_norm)
            gate = self.illum_gate(x_norm)  # the I branch is its own illumination guide
            x = x + gate * attn
            return x + self.gdfn(self.norm(x))  # same form as I_LCA

    HV.__name__, I.__name__ = f"IG_HV_LCA_{variant}", f"IG_I_LCA_{variant}"
    return HV, I


for _v in GATES:
    globals()[f"IG_HV_LCA_{_v}"], globals()[f"IG_I_LCA_{_v}"] = make_classes(_v)
