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

real_label = 1
fake_label = 0

################################ GENERATOR #################################################

class Generator(nn.Module):
    def __init__(self, nz=10, ngf=64, nc=1):
        """GAN generator.
        
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
    """Loss computed to train the GAN generator.

    Args:
      D: The discriminator whose forward function takes inputs of shape (batch_size, nc, 28, 28)
         and produces outputs of shape (batch_size, 1).
      fake_images of shape (batch_size, nc, 28, 28): Fake images produces by the generator.

    Returns:
      loss: The mean of the binary cross-entropy losses computed for all the samples in the batch.

    Notes:
    - Make sure that you process on the device given by `fake_images.device`.
    - Use values of global variables `real_label`, `fake_label` to produce the right targets.
    """

    loss = nn.BCELoss(reduction='mean')
    loss = loss.to(device)
    fake_images = fake_images.to(device)
    output = D(fake_images)
    output = output.to(device)
    label = torch.full((fake_images.shape[0],), real_label, device=device)
    label = label.to(device)
    l = loss(output, label)

    return l

    ########################## DISCRIMINATOR ##############################################

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        """GAN discriminator.
        
        Args:
          nc:  Number of channels in images.
          ndf: Base size (number of channels) of the discriminator layers.
        """
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc,ndf,kernel_size=4,stride=2,bias=False,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf,2*ndf,kernel_size=4,stride=2,bias=False,padding=1),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.2,inplace=True),
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*ndf,4*ndf,kernel_size=4,stride=2,bias=False,padding=1),
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(0.2,inplace=True),
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(4*ndf,nc,kernel_size=4,stride=2,bias=False,padding=1),
            nn.Sigmoid(),
        )
        
        self.to(device)

    def forward(self, x, verbose=False):
        """Classify given images into real/fake.
        
        Args:
          x of shape (batch_size, 1, 28, 28): Images to be classified.
        
        Returns:
          out of shape (batch_size,): Probabilities that images are real. All elements should be between 0 and 1.
        """
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

def discriminator_loss(D, real_images, fake_images):
    """Loss computed to train the GAN discriminator.

    Args:
      D: The discriminator.
      real_images of shape (batch_size, nc, 28, 28): Real images.
      fake_images of shape (batch_size, nc, 28, 28): Fake images produces by the generator.

    Returns:
      d_loss_real: The mean of the binary cross-entropy losses computed on the real_images.
      D_real: Mean output of the discriminator for real_images. This is useful for tracking convergence.
      d_loss_fake: The mean of the binary cross-entropy losses computed on the fake_images.
      D_fake: Mean output of the discriminator for fake_images. This is useful for tracking convergence.

    Notes:
    - Make sure that you process on the device given by `fake_images.device`.
    - Use values of global variables `real_label`, `fake_label` to produce the right targets.
    """
    
    loss = nn.BCELoss(reduction='mean')
    loss = loss.to(device)
    
    fake_images = fake_images.to(device)
    real_images = real_images.to(device)
    
    f_output = D(fake_images)
    f_output = f_output.to(device)
    
    r_output = D(real_images)
    r_output = r_output.to(device)    
    
    label = torch.full((fake_images.shape[0],), real_label, device=device)
    label = label.to(device)
    
    f_label = torch.full((fake_images.shape[0],), fake_label, device=device)
    f_label = f_label.to(device)
    
    d_loss_real = loss(r_output, label)
    
    d_loss_fake = loss(f_output,f_label)
    
    D_real = torch.mean(r_output)
    D_fake = torch.mean(f_output)

    return d_loss_real, D_real, d_loss_fake, D_fake
