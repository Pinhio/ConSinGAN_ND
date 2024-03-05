import torch.nn as nn

import models.modules as modules

class DiscriminatorND(nn.Module):
    '''ND Discriminator for SinGAN

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
    num_layers : int
        number of convolutional blocks for the discriminator
    dimension : int
        dimension to use
    batch_norm : bool
        use batch normalization
    cbam : bool
        use cbam attention mechanism
    '''
    def __init__(self, in_channel, out_channel, num_channel, kernel_size, padding, alpha, num_layers, dimension=3, batch_norm=True, cbam=True):
        super().__init__()

        # attention module for channel attention
        if cbam:
            self.channel_attention = modules.ChannelAttention(in_channel, dimension=dimension)

        # layer to be used for last convolution
        conv = nn.Conv3d if dimension == 3 else nn.Conv2d

        # first layer takes single channel input image
        self.head = modules.ConvolutionalBlockND(in_channel, num_channel, kernel_size, padding=padding, alpha=alpha, batch_norm=batch_norm, dimension=dimension)

        self.body = nn.Sequential()
        # minus 2 because of head and body
        for i in range(num_layers):
            block = modules.ConvolutionalBlockND(num_channel, num_channel, kernel_size, padding=padding, alpha=alpha, batch_norm=batch_norm, dimension=dimension)
            self.body.add_module(f'block_{i}', block)
        
        self.tail = conv(num_channel, out_channel, kernel_size, stride=1, padding=padding)

        # recursively apply initialization to every submodule
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d):
            module.weight.data.normal_(mean=1.0, std=0.02)
            module.bias.data.fill_(0)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x