import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from models.blocks import *


class SPADENorm(nn.Module):
    def __init__(self, norm_nc, ref_nc, norm_type='spade', ksz=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        mid_c = 64 

        self.norm_type = norm_type
        if norm_type == 'spade':
            self.conv1 = nn.Sequential(
                     nn.Conv2d(ref_nc, mid_c, ksz, 1, ksz//2),
                     nn.LeakyReLU(0.2, True),
                    )
            self.gamma_conv = nn.Conv2d(mid_c, norm_nc, ksz, 1, ksz//2)
            self.beta_conv = nn.Conv2d(mid_c, norm_nc, ksz, 1, ksz//2)
        
    def get_gamma_beta(self, x, conv, gamma_conv, beta_conv):
        act = conv(x)
        gamma = gamma_conv(act)
        beta = beta_conv(act)
        return gamma, beta 
      
    def forward(self, x, ref):
        normalized_input = self.param_free_norm(x)
        if x.shape[-1] != ref.shape[-1]:
            ref = nn.functional.interpolate(ref, x.shape[2:], mode='bicubic', align_corners=False)
        if self.norm_type == 'spade':
            gamma, beta = self.get_gamma_beta(ref, self.conv1, self.gamma_conv, self.beta_conv)
            return normalized_input * gamma + beta
        elif self.norm_type == 'in':
            return normalized_input


class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout, ref_nc, relu_type, norm_type='spade'):
        super().__init__()

        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
 
        # define normalization layers
        self.norm_0 = SPADENorm(fmiddle, ref_nc, norm_type) 
        self.norm_1 = SPADENorm(fmiddle, ref_nc, norm_type) 
        self.relu = ReluLayer(fmiddle, relu_type) 

    def forward(self, x, ref):
        res = self.conv_0(self.relu(self.norm_0(x, ref)))
        res = self.conv_1(self.relu(self.norm_1(res, ref)))
        out = x + res 

        return out


class PSFRGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, in_size=512, out_size=512, min_feat_size=16, ngf=64, n_blocks=9, parse_ch=19, relu_type='relu',
            ch_range=[32, 1024], norm_type='spade'):
        super().__init__()
        
        min_ch, max_ch = ch_range
        ch_clip = lambda x: max(min_ch, min(x, max_ch))
        get_ch = lambda size: ch_clip(1024*16//size)

        self.const_input = nn.Parameter(torch.randn(1, get_ch(min_feat_size), min_feat_size, min_feat_size)) 
        up_steps = int(np.log2(out_size//min_feat_size))
        self.up_steps = up_steps

        ref_ch = 19+3 

        head_ch = get_ch(min_feat_size)
        head = [
                nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1),
                SPADEResBlock(head_ch, head_ch, ref_ch, relu_type, norm_type),
                ]

        body = []
        for i in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2) 
            body += [
                    nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                        SPADEResBlock(cout, cout, ref_ch, relu_type, norm_type)
                        )
                    ]
            head_ch = head_ch // 2

        self.img_out = nn.Conv2d(ch_clip(head_ch), output_nc, kernel_size=3, padding=1)

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.upsample = nn.Upsample(scale_factor=2)
        
    def forward_spade(self, net, x, ref):
        for m in net:
            x = self.forward_spade_m(m, x, ref)
        return x

    def forward_spade_m(self, m, x, ref):
        if isinstance(m, SPADENorm) or isinstance(m, SPADEResBlock):
           x = m(x, ref)
        else:
           x = m(x)
        return x

    def forward(self, x, ref):
        b, c, h, w = x.shape
        const_input = self.const_input.repeat(b, 1, 1, 1)
        ref_input = torch.cat((x, ref), dim=1)
        
        feat = self.forward_spade(self.head, const_input, ref_input)

        for idx, m in enumerate(self.body):
            feat = self.forward_spade(m, feat, ref_input) 

        out_img = self.img_out(feat)

        return out_img 


if __name__ == '__main__':
    x = torch.randn(2, 16, 567, 234)
    nearest_interpolate(x)
