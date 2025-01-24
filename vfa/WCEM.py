import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pywt
from mmfewshot.detection.models import AGGREGATORS
from torch.autograd import Function


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


def wavelet_transform_init(filters):
    class WaveletTransform(Function):
        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = inverse_wavelet_transform(grad_output, filters)
            return grad, None

    return WaveletTransform().apply


def inverse_wavelet_transform_init(filters):
    class InverseWaveletTransform(Function):
        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = inverse_wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = wavelet_transform(grad_output, filters)
            return grad, None

    return InverseWaveletTransform().apply

@AGGREGATORS.register_module()
class WECM(nn.Module):
    def __init__(self, dim=1024, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.wtconv = WTConv2d(hidden_dim, hidden_dim)

    def forward(self, x):
        hh, ww = x.shape[2], x.shape[3]
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1).contiguous()
        x = torch.cat((x1, x2), dim=1).contiguous()
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=ww).contiguous()
        x = self.linear1(x)
        x_1, x_2 = x.chunk(2, dim=-1)
        x_1 = rearrange(x_1, 'b (h w) (c) -> b c h w', h=hh, w=ww).contiguous()
        x_1 = self.wtconv(x_1)
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c', h=hh, w=ww).contiguous()
        x = x_1 * x_2
        x = self.linear2(x)
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=hh, w=ww).contiguous()
        return x

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        padding = (kernel_size - 1) // 2
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, stride=1, dilation=1,
                                   groups=in_channels, bias=bias)

    def forward(self, x):
        x = self.wt_function(x)
        x = self.iwt_function(x)
        x = self.base_conv(x)
        return x