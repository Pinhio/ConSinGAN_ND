import torch
import torch.nn as nn

class ConvolutionalBlockND(nn.Module):
    '''ND Convolutional block for SinGAN

    Implementation based on Shaham et al. 2019 and Hinz et al. 2020

    Parameters
    ----------
    num_channel : int
        number of channels to be used in convolutional layers
    kernel_size : int
        kernel size for convolutional layers
    padding : int
        padding for convolutional layers
    alpha : float
        negative slope for leaky ReLU activation
    batch_norm : bool
        include batch norm or not
    dimension : int
        dimension to use (2 or 3)
    '''

    def __init__(self, in_channel, out_channel, kernel_size, padding, alpha, batch_norm=True, dimension=3):
        super().__init__()
        
        # get modules of fitting dimensionality
        conv = nn.Conv3d if dimension == 3 else nn.Conv2d
        batch = nn.BatchNorm3d if dimension == 3 else nn.BatchNorm2d

        self.block = nn.Sequential()
        self.block.add_module('conv', conv(in_channel, out_channel, kernel_size, stride=1, padding=padding))
        if batch_norm:
            self.block.add_module('batch_norm', batch(out_channel))
        self.block.add_module('lrelu', nn.LeakyReLU(alpha))
        
    def forward(self, x):
        return self.block(x)


# CBAM oriented on https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py
# original paper: https://link.springer.com/10.1007/978-3-030-01234-2_1
class ChannelAttention(nn.Module):

    def __init__(self, channel, reduction=16, bias=False, dimension=3):
        super().__init__()

        # pools and mlp definition
        self.avg_pool = nn.AdaptiveAvgPool3d(1) if dimension == 3 \
            else nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool3d(1) if dimension == 3 \
            else nn.AdaptiveMaxPool2d(1)

        # important, because in 3D case it could result in 0 channels
        reduced_channel = max(1, channel//reduction)
        conv = nn.Conv3d if dimension == 3 else nn.Conv2d

        self.mlp = nn.Sequential(
            conv(channel, reduced_channel, 1, bias=bias),
            nn.ReLU(),
            conv(reduced_channel, channel, 1, bias=bias)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgpool = self.avg_pool(x)
        maxpool = self.max_pool(x)

        avgpool = self.mlp(avgpool)
        maxpool = self.mlp(maxpool)

        out = self.sigmoid(avgpool+maxpool)
        return out
    

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7, dimension=3):
        super().__init__()

        conv = nn.Conv3d if dimension == 3 else nn.Conv2d
        
        # 2 input channels --> max and avg of channels
        self.convolution = conv(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        maxpool, _ = torch.max(x, dim=1, keepdim=True)
        avgpool = torch.mean(x, dim=1, keepdim=True)

        # concatenate the planes
        block = torch.cat([maxpool, avgpool], dim=1)
        block = self.convolution(block)
        out = self.sigmoid(block)
        return out
        
class CBAM(nn.Module):

    def __init__(self, channel, kernel_size=7, reduction=16, bias=False, dimension=3):
        super().__init__()
        self.channel_attention = ChannelAttention(channel, reduction, bias, dimension=dimension)
        self.spatial_attention = SpatialAttention(kernel_size, dimension=dimension)

    def forward(self, x):
        attention_map = x * self.spatial_attention(x)
        out = attention_map * self.channel_attention(attention_map)
        return out + x
