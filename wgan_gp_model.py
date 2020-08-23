import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

import tools
import tests

################################ GENERATOR #################################################
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        """WGAN generator.
        
        Args:
          nz:  Number of elements in the latent code.
          ngf: Base size (number of channels) of the generator layers.
          nc:  Number of channels in the generated images.
        """
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz,4*ngf,kernel_size=4,stride=2,bias=False,padding=1),
            nn.BatchNorm2d(4*ngf),
            nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(4*ngf,2*ngf,kernel_size=4,stride=2,bias=False,padding=0),
            nn.BatchNorm2d(2*ngf),
            nn.ReLU(),
        )
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(2*ngf,ngf,kernel_size=4,stride=2,bias=False,padding=0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
        )
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,bias=False,padding=1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, z, verbose=False):
        """Generate images by transforming the given noise tensor.
        
        Args:
          z of shape (batch_size, nz, 1, 1): Tensor of noise samples. We use the last two singleton dimensions
              so that we can feed z to the generator without reshaping.
          verbose (bool): Whether to print intermediate shapes (True) or not (False).
        
        Returns:
          out of shape (batch_size, nc, 28, 28): Generated images.
        """
        z = z.to(device)
        z = self.layer1(z)
        if verbose:
            print(z.shape)
        z = self.layer2(z)
        if verbose:
            print(z.shape)
        z = self.layer3(z)
        if verbose:
            print(z.shape)
        z = self.layer4(z)
        if verbose:
            print(z.shape)
        
        
        return z

def generator_loss(D, fake_images):
    """Loss computed to train the WGAN generator.

    Args:
      D: The critic whose forward function takes inputs of shape (batch_size, nc, 28, 28)
         and produces outputs of shape (batch_size, 1).
      fake_images of shape (batch_size, nc, 28, 28): Fake images produces by the generator.

    Returns:
      loss: The relevant part of the WGAN value function.
    """
    fake_images = fake_images.to(device)
    l = -D(fake_images).mean()

    return l

################################ DISCRIMINATOR #################################################

class Critic(nn.Module):
    def __init__(self, nc=1, ndf=64):
        """
        Args:
          nc:  Number of channels in the images.
          ndf: Base size (number of channels) of the critic layers.
        """
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc,ndf,kernel_size=4,stride=2,bias=False,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf,2*ndf,kernel_size=4,stride=2,bias=False,padding=1),
            nn.InstanceNorm2d(2*ndf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2,inplace=True),
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*ndf,4*ndf,kernel_size=4,stride=2,bias=False,padding=1),
            nn.InstanceNorm2d(4*ndf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2,inplace=True),
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(4*ndf,nc,kernel_size=4,stride=2,bias=False,padding=1),
            
            
        )

        
        
        self.to(device)

    def forward(self, x, verbose=False):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Images to be evaluated.
        
        Returns:
          out of shape (batch_size,): Critic outputs for images x.
        """
        x = x.to(device)
        x = self.layer1(x)
        if verbose:
            print(x.shape)
        x = self.layer2(x)
        if verbose:
            print(x.shape)
        x = self.layer3(x)
        if verbose:
            print(x.shape)
        x = self.layer4(x)
        x = x.view(x.shape[0])
        return x

def critic_loss(critic, real_images, fake_images):
    """
    Args:
      critic: The critic.
      real_images of shape (batch_size, nc, 28, 28): Real images.
      fake_images of shape (batch_size, nc, 28, 28): Fake images.

    Returns:
      loss (scalar tensor): Loss for training the WGAN critic.
    """
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)
    l = - torch.mean(critic(real_images)) + torch.mean(critic(fake_images))
    return l

def gradient_penalty(critic, real, fake_detached):
    """
    Args:
      critic: The critic.
      real of shape (batch_size, nc, 28, 28): Real images.
      fake_detached of shape (batch_size, nc, 28, 28): Fake images (detached from the computational graph).

    Returns:
      grad_penalty (scalar tensor): Gradient penalty.
      x of shape (batch_size, nc, 28, 28): Points x-hat in which the gradient penalty is computed.
    """
    critic = critic.to(device)
    real = real.to(device)
    fake_detached = fake_detached.to(device)
    batch_size = real.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real)
    alpha = alpha.to(device)
    
    interpolated = alpha * real+ (1 - alpha) * fake_detached
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)
    

    # Calculate probability of interpolated examples
    prob_interpolated = critic(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)


    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    grad_penalty = ((gradients_norm - 1) ** 2)
    return  grad_penalty,interpolated