import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import models.discriminators as dis
import models.generators as gen
from config_loader import Config
from models.functions import *


def train(img:np.ndarray, config:Config) -> None:
    '''Trains a SinGAN model based on a GrowingGeneratorND and given Image.

        Note: img has to be in (D x) H x W format.
        This is needed to comply with pytorch dimensions.

        Parameters
        ----------
        img : np.ndarray
            Image to use for training
        config : Config
            Configuration information
    '''

    # Always use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # convert image to tensor, compute scales
    # torch represents as N, C, (D,) H, W
    if config.dimension == 3:
        img_tensor = torch.reshape(torch.from_numpy(img).float(), [1,1,*img.shape]).to(device)
    elif config.dimension == 2:
        img_tensor = torch.reshape(torch.from_numpy(img).float(), [1,*img.shape]).to(device)
    else:
        raise NotImplementedError('Only 2 and 3 dimensional inputs can be processed.')
    
    img_tensor = rescale_to_max_size(img_tensor, config.max_size, dimension=config.dimension) # läuft in 3D
    scale_factor = get_scale_factor(img_tensor, config.min_size, config.num_scales)
    img_pyramid = get_image_pyramid(img_tensor, scale_factor, config.num_scales, dimension=config.dimension)

    # prepare filesystem
    config.save_dir = init_save_directory(config)

    # store config as json in output directory
    save_config(config)

    # init growing generator and noise variables
    generator = gen.GrowingGeneratorND(in_channel=img_pyramid[0].shape[1],
                                     out_channel=img_tensor.shape[1],
                                     num_channel=config.num_feature_channels,
                                     kernel_size=config.kernel_size,
                                     padding=config.padding,
                                     alpha=config.lrelu_alpha,
                                     num_layers=config.num_layers,
                                     dimension=config.dimension,
                                     cbam = config.cbam).to(device)
    
    fixed_noise = []
    noise_amp = []

    # start loop for training
    for curr_scale in range(config.num_scales):

        # init and store a directory for outputs of this scale
        config.curr_scale_dir = init_scale_directory(config, curr_scale)
        #print(config.curr_scale_dir)

        # init discriminator (new discriminator for every scale)
        discriminator = dis.DiscriminatorND(in_channel=img_pyramid[curr_scale].shape[1],
                                            out_channel=img_pyramid[curr_scale].shape[1],
                                            num_channel=config.num_feature_channels,
                                            kernel_size=config.kernel_size,
                                            padding=config.padding,
                                            alpha=config.lrelu_alpha,
                                            num_layers=config.num_layers,
                                            dimension=config.dimension,
                                            batch_norm=config.batch_norm,
                                            cbam=config.cbam).to(device)
        
        # init new stage of G and load params of D from last stage
        if curr_scale > 0:
            generator.init_next_scale()
            discriminator.load_state_dict(torch.load(f'{config.save_dir}/{curr_scale-1}/netD.pth'))

        # train scale
        generator, discriminator, fixed_noise, noise_amp = \
            train_single_scale(
                generator, discriminator, img_pyramid,
                fixed_noise, noise_amp, curr_scale, device, config
            )    
        
        # save networks
        save_networks(generator, discriminator, config.curr_scale_dir)
        del discriminator
    
    return

def train_single_scale(netG, netD, img_pyramid, fixed_noise, noise_amp, curr_scale, device, config):
    '''Trains a single scale of the SinGAN model.
    '''

    # get shapes of image pyramid and assign current image
    img_shape_pyramid = [i.shape for i in img_pyramid]
    curr_image = img_pyramid[curr_scale]

    # define z_opt for reconstruction training
    if curr_scale == 0:
        z_opt = img_pyramid[0]
    else:
        z_opt = generate_noise(img_shape_pyramid[curr_scale], config.num_feature_channels, device)

    fixed_noise.append(z_opt.detach())

    # define optimizers and schedulers
    betas = (config.beta1_adam, 0.999)
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=betas)
    parameter_list_G = netG.get_param_list(config.active_scales, config.lr, config.lr_factor)
    optimizerG = optim.Adam(parameter_list_G, lr=config.lr, betas=betas)

    # define learning rate schedules
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8 * config.epochs], gamma=config.gamma_scheduler)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8 * config.epochs], gamma=config.gamma_scheduler)

    # calculate noise amplification
    if curr_scale == 0:
        noise_amp.append(1)
    else:
        # needed for computation of z_reconstruction
        noise_amp.append(0)
        z_reconstruction = netG(fixed_noise, img_shape_pyramid, noise_amp)
        RMSE = torch.sqrt(F.mse_loss(z_reconstruction, curr_image)).detach()
        noise_amp[-1] = config.noise_amp * RMSE

    # start training
    epochs = tqdm(range(config.epochs))
    for epoch in epochs:
        epochs.set_description(f'scale [{curr_scale}/{config.num_scales-1}]')

        # noise for unconditional generation
        noise = generate_noise_pyramid(img_shape_pyramid, curr_scale, config, device)     

        # update D
        for step_idx in range(config.Dsteps):

            # train with real downsampled image of this scale
            netD.zero_grad()
            real_output = netD(curr_image)
            error_D_real = -real_output.mean()

            # train with fake
            if step_idx == config.Dsteps - 1:
                fake_image = netG(noise, img_shape_pyramid, noise_amp)
            else:
                with torch.no_grad():
                    fake_image = netG(noise, img_shape_pyramid, noise_amp)

            fake_output = netD(fake_image.detach())
            error_D_fake = fake_output.mean()

            # gradient penalty term
            gradient_penalty = calc_gradient_penalty(netD, curr_image, fake_image, config.lambda_grad_penalty, device)

            # total error
            error_D_total = error_D_real + error_D_fake + gradient_penalty
            error_D_total.backward()
            optimizerD.step()

        # update G
        fake_output = netD(fake_image)
        error_G = -fake_output.mean()

        # reconstruction loss
        reconstruction = netG(fixed_noise, img_shape_pyramid, noise_amp)
        reconstruction_loss = config.alpha_rec_loss * F.mse_loss(reconstruction, curr_image)

        # total error
        netG.zero_grad()
        error_G_total = error_G + reconstruction_loss
        error_G_total.backward()

        for _ in range(config.Gsteps):
            optimizerG.step()

        # show intermediate results and save sample images
        if epoch % 250 == 0 or epoch+1 == config.epochs:
            print(f'\nstage {curr_scale}, epoch {epoch+1}')
            print(f'Loss/train/D/real/{-error_D_real.item()}')
            print(f'Loss/train/D/fake/{error_D_fake.item()}')
            print(f'Loss/train/D/gradient_penalty/{gradient_penalty.item()}')
            print(f'Loss/train/G/gen/{error_G.item()}')
            print(f'Loss/train/G/reconstruction/{reconstruction_loss.item()}')
        if epoch % 500 == 0 or epoch+1 == config.epochs:
            save_image(f'{config.curr_scale_dir}/fake_sample_{epoch+1}.jpg', fake_image.detach())
            save_image(f'{config.curr_scale_dir}/reconstruction_{epoch+1}.jpg', reconstruction.detach())

    # schedulers
    schedulerD.step()
    schedulerG.step()

    return netG, netD, fixed_noise, noise_amp


