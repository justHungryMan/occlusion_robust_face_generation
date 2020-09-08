import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self, block, layer_count):
        super(Generator, self).__init__()

        # 256 * 256
        self.in_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, kernel_size=7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv256_128 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv128_64 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.residual = self.make_layer(block, layer_count, 256, 256)
            
        self.conv64_128 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv128_256 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1),
            nn.Tanh(),
        )

    def make_layer(self, block, layer_count, in_channels, out_channels):
        layers = []
        for i in range(layer_count):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.in_layer(x)
        out = self.conv256_128(out)
        out = self.conv128_64(out)
        out = self.residual(out)
        out = self.conv64_128(out)
        out = self.conv128_256(out)
        out = self.out_layer(out)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv256_128 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv128_64 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv64_32 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv32_16 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_layer = nn.Conv2d(512, 1, kernel_size=4, padding=1)
    
    def forward(self, x):
        out = self.conv256_128(x)
        out = self.conv128_64(out)
        out = self.conv64_32(out)
        out = self.conv32_16(out)
        out = self.out_layer(out)
        
        # Patch GAN
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
        return out
