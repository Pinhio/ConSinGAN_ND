# ConSinGAN_ND

ConSinGAN implementation for 2D and 3D images.

It is based mainly on the two papers:
- [Shaham et al. 2019](https://ieeexplore.ieee.org/document/9008787/)
- [Hinz et al. 2020](https://arxiv.org/abs/2003.11512)

The used attention mechanism was introduced by [Woo et al. 2018](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_1) and (to the best of my knowledge) first used in another SinGAN implementation by [Gu et al. 2021](https://www.mdpi.com/2072-4292/13/9/1713)

This project includes the model and trining pipeline for a n-dimensional ConSinGAN with an additional attention mechanism. 2D and 3D images can be used as input.

SinGan (Single Image Generative Adversarial Network) is a one-shot GAN architecture. This means, that only one single image is used for training. For implementation details, please refer to the inline documentation as well as the abovementiones papers.
