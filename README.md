# ConSinGAN_ND

**Note:** This project is an independent result of my masters thesis on 3D medical image data augmentation, which is still in work. Thus, it is not yet conveniently runnable with command line parameters but will be further updated. 

ConSinGAN implementation for 2D and 3D images.

It is primarily inspired by two papers:
- [Shaham et al. 2019](https://ieeexplore.ieee.org/document/9008787/)
- [Hinz et al. 2020](https://arxiv.org/abs/2003.11512)

The used attention mechanism was introduced by [Woo et al. 2018](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_1) and first used in another SinGAN implementation by [Gu et al. 2021](https://www.mdpi.com/2072-4292/13/9/1713)

This project includes the model and trining pipeline for a n-dimensional ConSinGAN with an additional attention mechanism. 2D and 3D images can be used as input.

SinGan (Single Image Generative Adversarial Network) is a one-shot GAN architecture. This means, that only one single image is used for training. For details on the achitecture, please refer to the inline documentation as well as the abovementiones papers.

## Training of a model

An example image is given in the `/data` directory. For execution of the training with the example image, just run `main.py`. If you wish to use your own image, you may add it to the `/data` directory and change the file name in `main.py`. Note that this network is a single image GAN and thus is only trained with a single image. It is also worth noting that the use of CUDA is highly recommended and that 3D computation takes considerably more time than 2D computation (but also yields better results).

The trained networks of each scale will be saved in the `/out` directory. In addition, every 500 epochs one fake sample and one reconstruction image are saved.

## Configuration

This section provides a small overview about the configuration of the Network.

| Parameter            | Default  | Short Description |
| :------------------- | :------- | :---------------- |
| data_dir             | 'data'   | directory with input data |
| out_dir              | 'out'    | output directory |
| dimension            | 3        | dimension of input image |
| epochs               | 2000     | number of epochs per scale |
| num_scales           | 8        | number of scales |
| active_scales        | 3        | number of concurrently trained scales |
| num_layers           | 3        | number of convolutional blocks per scale |
| num_feature_channels | 64       | number of feature channels in convolutional blocks |
| kernel_size          | 3        | kernel size for convolutional layers |
| padding              | 1        | padding for convolutional layers |
| batch_norm           | true     | use batch normalization within discriminator |
| cbam                 | true     | use CBAM attention module |
| lrelu_alpha          | 0.05     | alpha parameter od Leaky ReLU |
| beta1_adam           | 0.5      | first beta for adam optimizer |
| lr                   | 0.0005   | learning rate for Generator and Discriminator |
| lr_factor            | 0.1      | factor to reduce learning rate for concurrently trained scales |
| gamma_scheduler      | 0.1      | gamma parameter for schedulers |
| noise_amp            | 0.1      | factor for calculation of noise amplification |
| alpha_rec_loss       | 10       | alpha parameter of WGAN reconstruction loss formula |
| lambda_grad_penalty  | 0.1      | factor for gradient penalty term |
| Dsteps               | 3        | number of optimizer steps for discriminator per epoch |
| Gsteps               | 3        | number of optimizer steps for Generator per epoch |
| max_size             | 250      | maximum side length of input image |
| min_size             | 25       | minimum side length of downscaled image |
