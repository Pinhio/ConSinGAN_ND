from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F

import models.modules as modules

class GrowingGeneratorND(nn.Module):
    '''ND GrowingGenerator for SinGAN

    Implementation based on Shaham et al. 2019 and Hinz et al. 2020

    Parameters
    ----------
    in_channel : int
        number of input channels
    out_channel : int
        number of output channels
    num_channel : int
        number of channels to be used in convolutional layers
    kernel_size : int
        kernel size for convolutional layers
    padding : int
        padding for convolutional layers
    alpha : float
        negative slope for leaky ReLU activation
    num_layers : int
        number of convolutional blocks per scale
    dimension : int
        dimension to use
    cbam : bool
        use cbam attention mechanism
    '''

    def __init__(self, in_channel, out_channel, num_channel, kernel_size, padding, alpha, num_layers, dimension=3, cbam=True) -> None:
        super().__init__()

        # for final block
        self.conv = nn.Conv3d if dimension == 3 else nn.Conv2d
        # for initialization of weights
        self.batch = nn.BatchNorm3d if dimension == 3 else nn.BatchNorm2d
        # for upsampling in forward function
        self.interpolation_mode = 'trilinear' if dimension == 3 else 'bicubic'

        # initial block for channel adjustment
        self.head = modules.ConvolutionalBlockND(in_channel, num_channel, kernel_size, padding=padding, alpha=alpha, dimension=dimension)

        # Module dict for main part. Enables to keep track of parameter groups.
        self.body = nn.ModuleList({})
        scale_0 = nn.Sequential()
        for i in range(num_layers):
            block = modules.ConvolutionalBlockND(num_channel, num_channel, kernel_size, padding, alpha, dimension=dimension)
            scale_0.add_module(f'block_{i}', block)
        if cbam:
            scale_0.add_module('cbam', modules.CBAM(num_channel, dimension=dimension))
        self.body.append(scale_0)

        # final block for channel adjustments and normalization
        self.tail = nn.Sequential(
            self.conv(num_channel, out_channel, kernel_size, 1, padding),
            nn.Tanh()
        )

        # recursively apply initialization to every submodule
        self.apply(self._init_weights)

    def init_next_scale(self) -> None:
        self.body.append(deepcopy(self.body[-1]))

    def get_param_list(self, active_scales:int, lr:float, lr_factor:float) -> list:
        ''' Updates parameter groups and learning rate

        After Initialization of a new scale, this function serves for freezing
        the parameters of earlier scales and adjusting the learning rate of 
        the scales that are to be trained.

        Parameters
        ----------

        active_scales : int
            how many scales shall be included for learning
        lr : float
            learning rate of topmost scale
        lr_factor : float
            factor to be multiplied to lr for each inner stage
            that is trained
        '''
        for block in self.body[:-active_scales]:
            for param in block.parameters():
                param.requires_grad = False
        
        # set different learning rate for lower stages
        parameter_list = [{"params": block.parameters(), "lr": lr * (lr_factor**(len(self.body[-active_scales:])-1-idx))}
            for idx, block in enumerate(self.body[-active_scales:])]
        # set head (is also a conv block so it uses up a scale)
        if len(self.body) < active_scales:
            parameter_list += [{"params": self.head.parameters(), "lr": parameter_list[0]['lr']}]
        # set tail
        parameter_list += [{"params": self.tail.parameters()}]

        return parameter_list
    
    def _init_weights(self, module) -> None:
        if isinstance(module, self.conv):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, self.batch):
            module.weight.data.normal_(mean=1.0, std=0.02)
            module.bias.data.fill_(0)

    
    # ggf block padding wieder zufügen, könnte an den Rändern nötig sein?    
    def forward(self, x, real_shapes, noise_amp):
        noise = deepcopy(x)
        x = self.head(x[0])

        # we do some upsampling for training models for unconditional generation to increase
        # the image diversity at the edges of generated images
        # im notebook nicht nötig gewesen
        # x = self._upsample(x, size=[x.shape[2] + 2, x.shape[3] + 2, x.shape[4] + 2])

        # first block without residual because its just noise
        x_prev_out = self.body[0](x)

        # iterate trough remaining blocks (if any)
        for idx, block in enumerate(self.body[1:], 1):

            # x_prev_out upsampled for resisual connection
            x_prev_out_1 = F.interpolate(x_prev_out, size=real_shapes[idx][2:], mode=self.interpolation_mode, align_corners=True)
            
            # apply block upsampled prev image
            x_prev = block(x_prev_out_1 + noise[idx] * noise_amp[idx])

            # residual connection
            x_prev_out = x_prev + x_prev_out_1

        out = self.tail(x_prev_out)

        return out