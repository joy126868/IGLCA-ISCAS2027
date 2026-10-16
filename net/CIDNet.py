import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin


class ICorrectionHead(nn.Module):
    """
    Very small HVI-I correction head.

    It only predicts a bounded residual for the I channel.
    H and V channels are kept unchanged.
    The last convolution is zero-initialized so the model starts
    exactly from the original CIDNet output.
    """
    def __init__(self, hidden_channels=16):
        super(ICorrectionHead, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self._init_as_identity()

    def _init_as_identity(self):
        last = self.body[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, i_base, i_input):
        correction_input = torch.cat([i_base, i_input], dim=1)
        return self.body(correction_input)


class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 use_i_correction=True,
                 i_delta_scale=0.03,
                 return_dict=False
                 ):
        super(CIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        self.use_i_correction = use_i_correction
        self.i_delta_scale = i_delta_scale
        self.return_dict = return_dict

        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.trans = RGB_HVI()

        # V2: lightweight I-channel correction head.
        # Input: [I_base_from_CIDNet, I_input_from_low_light]
        # Output: raw delta_i, later bounded by tanh and i_delta_scale.
        self.i_correction = ICorrectionHead(hidden_channels=16)

    def forward(self, x, return_dict=None):
        if return_dict is None:
            return_dict = self.return_dict

        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)

        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        base_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        base_rgb = self.trans.PHVIT(base_hvi)

        if self.use_i_correction:
            hv_base = base_hvi[:, 0:2, :, :]
            i_base = base_hvi[:, 2:3, :, :]
            i_input = hvi[:, 2:3, :, :].to(dtypes)

            delta_i_raw = self.i_correction(i_base, i_input)
            delta_i = self.i_delta_scale * torch.tanh(delta_i_raw)

            i_final = torch.clamp(i_base + delta_i, 0.0, 1.0)
            output_hvi = torch.cat([hv_base, i_final], dim=1)
            output_rgb = self.trans.PHVIT(output_hvi)
        else:
            delta_i = torch.zeros_like(base_hvi[:, 2:3, :, :])
            output_hvi = base_hvi
            output_rgb = base_rgb

        if return_dict:
            return {
                "enhanced": output_rgb,
                "base": base_rgb,
                "enhanced_hvi": output_hvi,
                "base_hvi": base_hvi,
                "delta_i": delta_i,
            }

        return output_rgb

    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi
