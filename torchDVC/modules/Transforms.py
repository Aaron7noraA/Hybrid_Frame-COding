import torch
from torch import nn
from torch import Tensor
from compressai.layers import GDN, SwinTransformerBlock, PatchEmbed, PatchUnEmbed, window_partition
from compressai.models.utils import conv, deconv


class ResidualBlock(nn.Sequential):

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__(
            conv(num_filters, num_filters//2, 1, stride=1),
            nn.ReLU(inplace=True),
            conv(num_filters//2, num_filters//2, 3, stride=1),
            nn.ReLU(inplace=True),
            conv(num_filters//2, num_filters, 1, stride=1)
        )

    def forward(self, input):
        return input + super().forward(input)

class DCVC_ResBlock(nn.Sequential):

    def __init__(self, num_filters):
        super(DCVC_ResBlock, self).__init__(
            nn.ReLU(inplace=True),
            conv(num_filters, num_filters, 3, stride=1),
            nn.ReLU(inplace=True),
            conv(num_filters, num_filters, 3, stride=1)
        )
            
    def forward(self, input):
        return input + super().forward(input)

class GoogleAnalysisTransform(nn.Sequential):

    def __init__(self, in_channels, num_features, num_filters, kernel_size, downsample_8=False, no_input_layer=False):
        super(GoogleAnalysisTransform, self).__init__(
            conv(in_channels, num_filters, kernel_size, stride=2) if not no_input_layer else nn.Identity(),
            GDN(num_filters) if not no_input_layer else nn.Identity(),
            conv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_features, kernel_size, stride=2 if not downsample_8 else 1)
        )

class GoogleSynthesisTransform(nn.Sequential):

    def __init__(self, out_channels, num_features, num_filters, kernel_size, downsample_8=False, no_input_layer=False):
        super(GoogleSynthesisTransform, self).__init__(
            deconv(num_features, num_filters, kernel_size, stride=2 if not downsample_8 else 1),
            GDN(num_filters, inverse=True),
            deconv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            deconv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True) if not no_input_layer else nn.Identity(),
            deconv(num_filters, out_channels, kernel_size, stride=2) if not no_input_layer else nn.Identity()
        )

class GoogleHyperAnalysisTransform(nn.Sequential):

    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperAnalysisTransform, self).__init__(
            conv(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(num_filters, num_hyperpriors, kernel_size=5, stride=2)
        )

class GoogleHyperSynthesisTransform(nn.Sequential):

    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransform, self).__init__(
            deconv(num_hyperpriors, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(num_filters, num_filters * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(num_filters * 3 // 2, 2 * num_features, kernel_size=3, stride=1)
        )

class DCVCAnalysisTransform(nn.Sequential):

    def __init__(self, in_channels, num_features, num_filters, kernel_size):
        super(DCVCAnalysisTransform, self).__init__(
            conv(in_channels, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            DCVC_ResBlock(num_filters),
            conv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            DCVC_ResBlock(num_filters),
            conv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_features, kernel_size, stride=2)
        )

class DCVCSynthesisTransform(nn.Sequential):

    def __init__(self, out_channels, num_features, num_filters, kernel_size):
        super(DCVCSynthesisTransform, self).__init__(
            deconv(num_features, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            deconv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            DCVC_ResBlock(num_filters),
            deconv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            DCVC_ResBlock(num_filters),
            deconv(num_filters, out_channels, kernel_size, stride=2)
        )

class MaskLayer(nn.Module):

    def __init__(self, p) -> None:
        super(MaskLayer, self).__init__()
        self.p = p
        self.mask_value = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x) -> Tensor:
        print("X: ", x.shape)
        if self.training:
            mask = torch.rand_like(x) < self.p
            x = x.clone()
            x[mask] = self.mask_value

        return x

''' Swin Transformer-based '''
class TokenEmbedding(nn.Module):
    
    def __init__(self, inc, embed_c, ws=2) -> None:
        super(TokenEmbedding, self).__init__()
        # self.unshuffle = nn.PixelUnshuffle(ws)
        # self.project = nn.Linear(inc * (ws**2), embed_c)
        self.project = nn.Sequential(
            conv(inc, embed_c, 3, stride=2),
            nn.LeakyReLU(inplace=True),
            conv(embed_c, embed_c, 3, stride=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # x = self.unshuffle(x)
        # x = self.project(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.project(x)
        return x
        

class TokenUnEmbedding(nn.Module):
    
    def __init__(self, embed_c, output_c, ws=2) -> None:
        super(TokenUnEmbedding, self).__init__()
        # self.project = nn.Linear(embed_c, output_c * (ws**2))
        # self.shuffle = nn.PixelShuffle(ws)
        self.project = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            deconv(embed_c, output_c, 3, stride=2),
            nn.LeakyReLU(inplace=True),
            conv(output_c, output_c, 3, stride=1)
        )

    def forward(self, x):
        # x = self.project(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x = self.shuffle(x)
        x = self.project(x)
        return x
        

class WindowEmbed(nn.Module):

    def __init__(self, dim, window_size, shift_size=0) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.patch_embed = PatchEmbed()

    def forward(self, x, x_size):
        H, W = x_size
        x = self.patch_embed(x)
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return x_windows

class SwinGeneralResidual(nn.Module):

    def __init__(self, depth, num_filters, input_resolution, num_heads, window_size, 
                 mlp_ratio, cross=True, cancel_out=True) -> None:
        super().__init__()
        self.cross_swin = SwinTransformerBlock(num_filters, input_resolution, num_heads, window_size, 
                            mlp_ratio=mlp_ratio, cross=cross, cancel_out=cancel_out)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=num_filters, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio)
            for i in range(depth)]) 
        self.norm1 = nn.LayerNorm(num_filters)
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()
    
    def forward(self, x, k, v, x_size):
        embed = self.patch_embed(x)
        tokenc = self.cross_swin(embed, x_size, k=k, v=v)
        token = tokenc
        for block in self.blocks:
            token = block(token, x_size)
        token = self.norm1(token)
        return self.patch_unembed(token, x_size), tokenc

class SwinGeneralSum(nn.Module):

    def __init__(self, depth, num_filters, input_resolution, num_heads, window_size, 
                 mlp_ratio, cross=True, cancel_out=True) -> None:
        super().__init__()
        self.cross_swin = SwinTransformerBlock(num_filters, input_resolution, num_heads, window_size, 
                            mlp_ratio=mlp_ratio, cross=cross, cancel_out=cancel_out)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=num_filters, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio)
            for i in range(depth)]) 
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()
    
    def forward(self, x, k, v, x_size):
        embed = self.patch_embed(x)
        token = self.cross_swin(embed, x_size, k=k, v=v)
        for block in self.blocks:
            token = block(token, x_size)
        return self.patch_unembed(token, x_size)