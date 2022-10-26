# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import CONV_LAYERS, kaiming_init, normal_init
from torch.nn import functional as F
from torch.nn.modules.utils import _triple


def conv3v(input, weight, stride, padding, dilation, groups):
    weight = weight.squeeze(2)  # (Cout, Cin, 1, K, K) -> (Cout, Cin, K, K)
    padding_hw = (0, padding, padding)
    padding_tw = (padding, 0, padding)
    padding_th = (padding, padding, 0)
    hw = F.conv3d(input, weight.unsqueeze(2), None, stride, padding_hw,
                  dilation, groups)
    tw = F.conv3d(input, weight.unsqueeze(3), None, stride, padding_tw,
                  dilation, groups)
    th = F.conv3d(input, weight.unsqueeze(4), None, stride, padding_th,
                  dilation, groups)
    return hw, tw, th


@CONV_LAYERS.register_module()
class CoSTa(nn.Module):
    """CoST(a) module.

    https://arxiv.org/abs/1903.01197.

    Args:
        in_channels (int): Same as nn.Conv3d.
        out_channels (int): Same as nn.Conv3d.
        kernel_size (int): Same as nn.Conv3d.
        stride (int | tuple[int]): Same as nn.Conv3d.
        padding (int): Same as nn.Conv3d.
        dilation (int | tuple[int]): Same as nn.Conv3d.
        groups (int): Same as nn.Conv3d.
        bias (bool): Same as nn.Conv3d.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _triple(stride)
        self.padding = padding
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = 'zeros'
        self.output_padding = (0, 0, 0)
        self.transposed = False

        self.alpha = nn.Parameter(torch.empty(3, out_channels))
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 1, kernel_size,
                        kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.alpha[0], 0.7)
        nn.init.constant_(self.alpha[1:], -0.7)
        kaiming_init(self)

    def forward(self, input):
        hw, tw, th = conv3v(input, self.weight, self.stride, self.padding,
                            self.dilation, self.groups)
        alpha = torch.softmax(self.alpha, dim=0)
        alpha = alpha.view(*alpha.shape, 1, 1, 1)
        output = hw * alpha[0] + tw * alpha[1] + th * alpha[2]

        if self.bias is not None:
            output += self.bias.view(*self.bias.shape, 1, 1, 1)

        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0, ) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


@CONV_LAYERS.register_module()
class CoSTb(nn.Module):
    """CoST(b) module.

    https://arxiv.org/abs/1903.01197.

    Args:
        in_channels (int): Same as nn.Conv3d.
        out_channels (int): Same as nn.Conv3d.
        kernel_size (int): Same as nn.Conv3d.
        stride (int | tuple[int]): Same as nn.Conv3d.
        padding (int): Same as nn.Conv3d.
        dilation (int | tuple[int]): Same as nn.Conv3d.
        groups (int): Same as nn.Conv3d.
        bias (bool): Same as nn.Conv3d.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _triple(stride)
        self.padding = padding
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = 'zeros'
        self.output_padding = (0, 0, 0)
        self.transposed = False

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 1, kernel_size,
                        kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc1 = nn.Linear(out_channels, out_channels, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)

        self.init_weights()

    def init_weights(self):
        kaiming_init(self)
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2, std=0.01)

    def forward(self, input):
        hw, tw, th = conv3v(input, self.weight, self.stride, self.padding,
                            self.dilation, self.groups)

        pool_hw = self.max_pool(hw).view(-1, self.out_channels)  # (N, C)
        pool_tw = self.max_pool(tw).view(-1, self.out_channels)  # (N, C)
        pool_th = self.max_pool(th).view(-1, self.out_channels)  # (N, C)

        x = torch.concat((pool_hw, pool_tw, pool_th), dim=0)  # (3N, C)
        x = self.fc1(x)  # (3N, C)
        x = x.view(3, -1).permute((1, 0))  # (N*C, 3)
        x = self.fc2(x)  # (N*C, 3)
        alpha = x.permute((1, 0)).view(  # noqa
            3, -1, self.out_channels, 1, 1, 1)  # (3, N, C, 1, 1, 1)
        alpha = torch.softmax(alpha, dim=0)

        output = hw * alpha[0] + tw * alpha[1] + th * alpha[2]

        if self.bias is not None:
            output += self.bias.view(*self.bias.shape, 1, 1, 1)

        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0, ) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
