import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.utils as utils

import tools

from scorer import Scorer

from data import trainloader

from dcgan_model import Generator, generator_loss, Discriminator, discriminator_loss



def main():
    """
    function to train model, plot generated samples, compute training score,
    save train model, load train model, and evaluate model
    """
    # device = torch.device('cuda:0')
    device = torch.device('cpu')

    skip_training=False
    batch_size = 100
    n_epochs = 20

    scorer = Scorer()
    scorer.to(device)

    nz = 10
    netG = Generator(nz=nz, ngf=64, nc=1)
    netD = Discriminator(nc=1, ndf=64)

    netD = netD.to(device)
    netG = netG.to(device)

    if not skip_training:
        d_optimizer = torch.optim.Adam(netD.parameters(),lr=0.0002,betas=(0.5, 0.999))
        g_optimizer = torch.optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            for i, data in enumerate(trainloader, 0):
                images, _= data
                images= images.to(device)


                netD.train()
                netD.zero_grad()
                d_optimizer.zero_grad()
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_images = netG(noise)
                d_loss_real, D_real, d_loss_fake, D_fake = discriminator_loss(netD, images, fake_images)
                d_loss_real.backward(retain_graph=True)
                d_loss_fake.backward(retain_graph=True)
                d_loss = d_loss_real + d_loss_fake
                d_optimizer.step()

                netG.train()
                netG.zero_grad()
                g_optimizer.zero_grad()
                g_loss = generator_loss(netD, fake_images)
                g_loss.backward(retain_graph=True)
                g_optimizer.step()

                
            with torch.no_grad():
            # Plot generated images
                z = torch.randn(144, nz, 1, 1, device=device)
                samples = netG(z)
                tools.plot_generated_samples(samples)

            # Compute score
                z = torch.randn(1000, nz, 1, 1, device=device)
                samples = netG(z)
                samples = (samples + 1) / 2  # Re-normalize to [0, 1]
                score = scorer(samples)
        

            print('Train Epoch {}: D_real {}: D_fake{}: score {}'.format(epoch +1,D_real,D_fake,score))   

        tools.save_model(netG, '11_dcgan_g.pth')
        tools.save_model(netD, '11_dcgan_d.pth')
    else:
        nz = 10
        netG = Generator(nz=nz, ngf=64, nc=1)
        netD = Discriminator(nc=1, ndf=64)

        tools.load_model(netG, '11_dcgan_g.pth', device)
        tools.load_model(netD, '11_dcgan_d.pth', device)   

        with torch.no_grad():
            z = torch.randn(1000, nz, 1, 1, device=device)
            samples = (netG(z) + 1) / 2
            score = scorer(samples)

        print(f'The trained DCGAN achieves a score of {score:.5f}')

if __name__ == "__main__":
    main()
