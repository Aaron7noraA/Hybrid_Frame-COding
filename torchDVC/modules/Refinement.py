import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import getfullargspec
from util.warp import torch_warp as warp


class ResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters, activation=nn.ReLU):
        inplace = 'inplace' in getfullargspec(activation.__init__).args
        super(ResidualBlock, self).__init__(
            activation(inplace=False) if inplace else activation(),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            activation(inplace=True) if inplace else activation(),
            nn.Conv2d(num_filters, num_filters, 3, padding=1)
        )

    def forward(self, input):
        return input + super().forward(input)

class DeepResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters, activation=nn.ReLU):
        super(DeepResidualBlock, self).__init__(
            ResidualBlock(num_filters, activation),
            ResidualBlock(num_filters, activation)
        )

    def forward(self, input):
        return input + super().forward(input)

class SuperDeepResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters, activation=nn.ReLU):
        super(SuperDeepResidualBlock, self).__init__(
            DeepResidualBlock(num_filters, activation),
            DeepResidualBlock(num_filters, activation)
        )

    def forward(self, input):
        return input + super().forward(input)

class DCVC_GS(nn.Module):

    def __init__(self, in_ch, h_ch, out_ch) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, 3, 1, 1),
            ResidualBlock(h_ch),
            ResidualBlock(h_ch),
            nn.Conv2d(h_ch, out_ch, 3, 1, 1)
        )

    def forward(self, ins):
        return self.net(torch.cat(ins, dim=1))

class GS(nn.Module):

    def __init__(self, in_ch, h_ch) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(h_ch, h_ch, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(h_ch, 3, 3, 1, 1)
        )

    def forward(self, ins):
        return self.net(torch.cat(ins, dim=1))

class Refinement(nn.Module):
    """Refinement UNet"""

    def __init__(self, in_channels, num_filters, out_channels=3):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
        self.l2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )

        self.d3 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d1 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, out_channels, 3, padding=1)
        )

    def forward(self, *input):
        if len(input) == 1:
            input = input[0]
        else:
            input = torch.cat(input, dim=1)

        conv1 = self.l1(input)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(deconv3 + conv2)
        deconv1 = self.d1(deconv2 + conv1)

        return deconv1

class ShrinkRefinement(nn.Module):
    """ShrinkRefinement UNet"""

    def __init__(self, in_channels, io_num_filters, num_filters, out_channels=3):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, io_num_filters, 3, padding=1),
            ResidualBlock(io_num_filters)
        )
        self.l2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(io_num_filters, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )

        self.d3 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.Conv2d(num_filters, io_num_filters, 3, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d1 = nn.Sequential(
            ResidualBlock(io_num_filters),
            nn.Conv2d(io_num_filters, io_num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(io_num_filters, out_channels, 3, padding=1)
        )

    def forward(self, *input):
        if len(input) == 1:
            input = input[0]
        else:
            input = torch.cat(input, dim=1)

        conv1 = self.l1(input)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(deconv3 + conv2)
        deconv1 = self.d1(deconv2 + conv1)

        return deconv1

class DeepRefinement(ShrinkRefinement):
    """DeepRefinement UNet"""

    def __init__(self, in_channels, io_num_filters, num_filters, out_channels=3):
        super(ShrinkRefinement, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, io_num_filters, 3, padding=1)
        )
        self.l2 = nn.Sequential(
            ResidualBlock(io_num_filters, nn.LeakyReLU),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(io_num_filters, num_filters, 3, padding=1),
            ResidualBlock(num_filters, nn.LeakyReLU)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            DeepResidualBlock(num_filters, nn.LeakyReLU)
        )

        self.d3 = nn.Sequential(
            DeepResidualBlock(num_filters, nn.LeakyReLU),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters, nn.LeakyReLU),
            nn.Conv2d(num_filters, io_num_filters, 3, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResidualBlock(io_num_filters, nn.LeakyReLU)
        )
        self.d1 = nn.Sequential(
            nn.Conv2d(io_num_filters, io_num_filters, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(io_num_filters, out_channels, 1)
        )

class Augment(nn.Module):
    """ Augment UNet """

    def __init__(self, in_channels, num_filters, out_channels=3):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
       
        self.d1 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, out_channels, 3, padding=1)
        )

    def forward(self, *input):
        if len(input) == 1:
            input = input[0]
        else:
            input = torch.cat(input, dim=1)

        conv1 = self.l1(input)
        deconv1 = self.d1(conv1)

        return deconv1

class ConvGRU2D(nn.Module):

    def __init__(self, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size, padding=kernel_size//2)
        self.in_ = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, x, hidden):
        g = self.gate(torch.cat([x, hidden], dim=1))
        z, r = torch.sigmoid(g).split(self.hidden_dim, dim=1)
        ht = torch.tanh(self.in_(torch.cat([x, r * hidden], dim=1)))
        new_hidden = (1. - z) * hidden + z * ht # we can't set hidden state in the forward when using DataParallel
        return new_hidden

    def init_hidden(self, B, H, W):
        return torch.zeros(B, self.hidden_dim, H, W, device=self.gate.weight.device, requires_grad=False)

class RNNRefinement(ShrinkRefinement):
    """RNNRefinement UNet"""

    def __init__(self, in_channels, io_num_filters, num_filters, out_channels=3):
        super().__init__(in_channels, io_num_filters, num_filters, out_channels)
        self.gru = ConvGRU2D(io_num_filters, 3)

    def init_hidden(self, B, H, W):
        return self.gru.init_hidden(B, H, W)

    def forward(self, hidden, flow, *input):
        if len(input) == 1:
            input = input[0]
        else:
            input = torch.cat(input, dim=1)

        hidden = warp(hidden, flow)

        conv1 = self.l1(input)
        conv1 = hidden = self.gru(conv1, hidden)

        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(deconv3 + conv2)
        deconv1 = self.d1(deconv2 + conv1)

        return deconv1, hidden


class FlowBlock(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(8, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=7, padding=3)
        )

    def forward(self, flow, im1, im2):
        scale = flow.size(0) / im2.size(0)
        flow_coarse = F.interpolate(flow, size=im2.size()[2:], mode='bilinear', align_corners=False) / scale
        res_coarse = super().forward(torch.cat([im1, warp(im2, flow_coarse), flow_coarse], dim=1)) - flow_coarse
        res = F.interpolate(res_coarse, size=flow.size()[2:], mode='bilinear', align_corners=False) * scale
        flow_fine = res + flow
        return flow_fine


class FlowRefinement(nn.Module):
    """FlowRefinement"""

    def __init__(self, level=4):
        super(FlowRefinement, self).__init__()
        self.level = level
        self.Blocks = nn.ModuleList([FlowBlock() for _ in range(level+1)])

    def forward(self, im2, im1, flow):  # backwarp
        # B, 6, H, W
        # im2 - source, im1 - target
        volume = [torch.cat([im1, im2], dim=1)]
        for _ in range(self.level):
            volume.append(F.avg_pool2d(volume[-1], kernel_size=2))

        for l, layer in enumerate(self.Blocks):
            flow = layer(flow, *volume[self.level-l].chunk(2, 1))

        return flow

