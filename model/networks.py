import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu, instance_norm

from .LARN import LARN


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,):
        super(GatedConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.sigmoid = torch.nn.Sigmoid()
        self.larn = LARN(out_channels)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        gated_m = self.gated(mask)
        x = elu(x) * gated_m
        x = self.larn(x, gated_m)
        x = instance_norm(x)
        return x
