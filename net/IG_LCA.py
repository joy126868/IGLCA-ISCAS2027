import torch
import torch.nn as nn

from net.transformer_utils import LayerNorm
from net.LCA import CAB, IEL


class IlluminationGate(nn.Module):
    """
    Lightweight spatial-channel gate for illumination-guided cross attention.

    The last convolution is zero-initialized and the output gate is computed as
    2 * sigmoid(logits), so the initial scale is exactly 1.0. This makes the
    IG-LCA block start close to the author's original LCA behavior instead of
    destroying CIDNet at the beginning of training.
    """
    def __init__(self, dim, reduction=4):
        super(IlluminationGate, self).__init__()
        hidden = max(dim // reduction, 8)
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(hidden, dim, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self._init_identity()

    def _init_identity(self):
        last = self.body[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, guide):
        return 2.0 * torch.sigmoid(self.body(guide))


class IG_HV_LCA(nn.Module):

    def __init__(self, dim, num_heads, bias=False):
        super(IG_HV_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.gdfn = IEL(dim)
        self.illum_gate = IlluminationGate(dim)

    def forward(self, x, y):
        x_norm = self.norm(x)
        y_norm = self.norm(y)
        attn = self.ffn(x_norm, y_norm)
        gate = self.illum_gate(y_norm)
        x = x + gate * attn

        # Same form as the author's HV_LCA: the IEL output becomes the updated feature.
        x = self.gdfn(self.norm(x))
        return x

class IG_I_LCA(nn.Module):

    def __init__(self, dim, num_heads, bias=False):
        super(IG_I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.gdfn = IEL(dim)
        self.illum_gate = IlluminationGate(dim)

    def forward(self, x, y):
        x_norm = self.norm(x)
        y_norm = self.norm(y)
        attn = self.ffn(x_norm, y_norm)
        gate = self.illum_gate(x_norm)
        x = x + gate * attn

        # Same form as the author's I_LCA.
        x = x + self.gdfn(self.norm(x))
        return x


