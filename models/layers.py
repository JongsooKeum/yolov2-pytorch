import torch
from torch import nn
import numpy as np

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias)

def max_pool(kernel_size, stride, padding=0):
    return nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=nn.LeakyReLU()):
        super().__init__()

        self.conv = conv2d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x):
        if self.act is None:
            return self.bn(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()

        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output