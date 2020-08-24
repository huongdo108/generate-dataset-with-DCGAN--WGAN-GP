# Generate dataset with Generative adversarial networks (GANs): Deep Convolutional GAN (DCGAN) and Wasserstein GAN with gradient penalty (WGAN-GP)

<img src="https://render.githubusercontent.com/render/math?math=\begin{align}
z &\sim N(0, I)
\\
x &= G(z)
\end{align}">

## Overview
The goal of this repository is to get familiar with generative adversarial networks and specifically DCGAN and WGAN-GP. GANs are models from which we can draw samples that will have a distribution similar to the distribution of the training data.


DCGAN was proposed by [Radford et al., 2015](https://arxiv.org/pdf/1511.06434.pdf).. WGAN-GP was proposed by  [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf).

## Dataset 
MNIST digits data from torchvision.datasets

## Deep Convolutional GAN (DCGAN)
DCGAN architecture includes 2 main components: Generator and Discriminator

**Generator**

The generative model that I use is:
<img src="https://github.com/huongdo108/generate-dataset-with-DCGAN--WGAN-GP/blob/master/images/dcgan_generator.PNG" align="centre">

The data is generated by applying a nonlinear transformation to samples drawn from the standard normal distribution.

**G** is modeled with a deep neural network. In DCGAN, the generator is made of only transposed convolutional layers `ConvTranspose2d` followed by `tanh`. 
The `tanh` nonlinearity guarantees that the output is between -1 and 1 which holds for our scaling of the training data.

**Loss for training the generator**

The generative model will be guided by a discriminator whose task is to separate (classify) data into two classes:
* true data (samples from the training set)
* generated data (samples generated by the generator).

The task of the generator is to confuse the discriminator as much as possible, which is the case when the distribution produced by the generator perfectly replicates the data distribution. Thus, a loss function is implemented to train the generator. The loss is the `binary_cross_entropy` loss computed with `real_label` as targets for the generated samples.

**Discriminator**

In DCGAN, the discriminator is a stack of only convolutional layers.

**Loss for training the discriminator**

The discriminator is trained to solve a binary classification problem: to separate real data from generated samples. Thus, the output of the discriminator should be a scalar between 0 and 1. 

A loss function is implemented to train the discriminator. The dicriminator uses the `binary_cross_entropy` loss,  `real_label` as targets for real samples and `fake_label` as targets for generated samples.

## Wasserstein GAN with gradient penalty (WGAN-GP)

WGAN-GP architecture includes 2 main components: Generator and Critic


<img src="https://github.com/huongdo108/generate-dataset-with-DCGAN--WGAN-GP/blob/master/images/wgan.PNG" align="centre">

**Generator**
The same architecture with DCGAN's generator

**Loss for training the generator**

The generator is trained to minimize the relevant part of the value function using a fixed critic **D**:

<img src="https://github.com/huongdo108/generate-dataset-with-DCGAN--WGAN-GP/blob/master/images/wgan_g_loss.PNG" align="centre">

**Critic**

In WGAN-GP, the discriminator is called a critic because it is not trained to classify. 

**Loss for training the WGAN critic**

<img src="https://github.com/huongdo108/generate-dataset-with-DCGAN--WGAN-GP/blob/master/images/wgan_c_loss.PNG" align="centre">

**Gradient penalty**

<img src="https://github.com/huongdo108/generate-dataset-with-DCGAN--WGAN-GP/blob/master/images/wgan_penalty.PNG" align="centre">

