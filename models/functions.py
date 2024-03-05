import os
import datetime
import json
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config_loader import Config


# functions for rescaling and image pyramid

def rescale_to_max_size(img:torch.Tensor, max_size:int, dimension:int=3) -> torch.Tensor:
    '''Rescale image to a given maximum side length 
    '''
    mode = 'trilinear' if dimension == 3 else 'bilinear'
    scale_factor = min(max_size / max(img.shape[-2:]), 1)
    rescaled = F.interpolate(img, scale_factor=scale_factor, mode=mode, align_corners=True)
    return rescaled

def get_scale_factor(img:torch.Tensor, min_size:int, num_scales:int) -> float:
    ''' Calculates scale factor for ALREADY RESIZED img

    Prameters
    ---------

    img : torch.Tensor
        base image
    min_size : int
        minimum image size on coarsest scale
    num_scales : int
        number of scales to compute
    '''
    base = min_size / min(img.shape[-2:])
    exp = 1 / (num_scales - 1)
    return pow(base, exp)

def get_image_pyramid(img:torch.Tensor, scale_factor:float, num_scales:int, dimension:int=3) -> list:
    '''Get pyramid of rescaled images

    These represent the resolution of the scales in the SinGAN

    Parameters
    ----------

    img : torch.Tensor
        image to rescale
    scale_factor : float
        factor for scaling of the dimensions
    num_scales : int
        number of scales in the pyramid
    '''
    mode = 'trilinear' if dimension == 3 else 'bilinear'
    pyramid = []
    for i in range(num_scales - 1):
        downscaled = F.interpolate(img, 
                                   scale_factor=pow(scale_factor, num_scales-i-1),
                                   mode=mode,
                                   align_corners=True)
        pyramid.append(downscaled)
    pyramid.append(img)
    return pyramid


# file system operations

def init_save_directory(config:Config, name:str=None) -> str:
    '''Create a sub directory of out_dir for the training
    '''
    # get name
    if name is None:
        save_dir = f"{config.out_dir}/{re.sub('[^0-9]+', '_', str(datetime.datetime.now()))}"
    else:
        save_dir = f"{config.out_dir}/{name}"
    # create directory
    try:
        os.makedirs(save_dir)
    except OSError:
        print(OSError)
        pass
    return save_dir

def init_scale_directory(config:Config, curr_scale:int) -> str:
    '''Create sub directory for a scale
    '''
    scale_dir = f'{config.save_dir}/{curr_scale}'
    try:
        os.makedirs(scale_dir)
    except OSError:
        print(OSError)
        pass
    return scale_dir


def save_networks(netG:nn.Module, netD:nn.Module, curr_scale_dir:str) -> None:
    '''Saves networks to belonging scale directory
    '''
    torch.save(netG.state_dict(), f'{curr_scale_dir}/netG.pth')
    torch.save(netD.state_dict(), f'{curr_scale_dir}/netD.pth')
    
def save_noises(noise:torch.Tensor, curr_scale_dir:str) -> None:
    '''Saves noise to belonging scale directory
    '''
    torch.save(noise, f'{curr_scale_dir}/z_opt.pth')

def save_image(name:str, img:torch.Tensor) -> None:
    '''Saves an image as numpy to save directory
    '''
    np.save(name, np.reshape(img.cpu().numpy(), img.shape[-3:]))

def save_config(config:Config) -> None:
    '''Saves config to save directory
    '''
    with open(f'{config.save_dir}/config.json', 'w') as f:
        json.dump(config.__dict__, f)


# functions for noise
    
def generate_noise(shape:torch.Size, num_channels:int, device:str) -> torch.Tensor:
    '''Generates a noise tensor

    Has the same shape as the referring image tensor but
    an adjusted amount of channels for higher scales.
    This is because scale 1 to n need a higher number of
    feature channels than the 0 scale.
    '''
    shape = list(shape)
    shape[1] = num_channels
    return torch.randn(*shape, device=device)

def generate_noise_pyramid(img_shape_pyramid:list, 
                           curr_scale:int, 
                           config:Config, 
                           device:str) -> list:
    '''Generate a pyramid of noise oriented on the pyramid shapes
    '''
    noise_pyramid = []
    for i in range(curr_scale+1):
        if i == 0:
            noise_pyramid.append(generate_noise(img_shape_pyramid[i],
                                                img_shape_pyramid[i][1],
                                                device).detach())
        else:
            noise_pyramid.append(generate_noise(img_shape_pyramid[i],
                                                config.num_feature_channels,
                                                device).detach())
    return noise_pyramid


# gradient penalty, adapted from Hinz et al. 2020
def calc_gradient_penalty(netD, real_data, fake_data, lambda_grad_pen, device):

    alpha = torch.rand(*(1, 1), device=device)
    alpha = alpha.expand(real_data.shape)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.shape, device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_grad_pen

    return gradient_penalty