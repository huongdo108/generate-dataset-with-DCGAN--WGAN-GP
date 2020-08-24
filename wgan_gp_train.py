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

from wgan_gp_model import Generator,generator_loss, Critic, critic_loss, gradient_penalty




def main():
    """
    function to train model, plot generated samples, compute training score,
    save train model, load train model, and evaluate model
    """

    # device = torch.device('cuda:0')
    device = torch.device('cpu')

    batch_size=32
    n_epochs = 15
    lambda_n = 10

    scorer = Scorer()
    scorer.to(device)
    
    nz = 10
    netG = Generator(nz=nz, ngf=128, nc=1).to(device)
    netD = Critic(nc=1, ndf=128).to(device)

    if not skip_training:
        d_optimizer = torch.optim.Adam(netD.parameters(),lr=0.0001)
        g_optimizer = torch.optim.Adam(netG.parameters(),lr=0.0001)


        for epoch in range(n_epochs):
            for i, data in enumerate(trainloader, 0):
                images, _= data
                images= images.to(device)


                netD.train()
                netD.zero_grad()
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_images = netG(noise)
                d_loss = critic_loss(netD, images, fake_images)
                
                grad_penalty,x_hat = gradient_penalty(netD, images, fake_images.detach())
                
                critic_loss_total = d_loss + grad_penalty*lambda_n
                critic_loss_total.sum().backward(retain_graph=True)
                d_optimizer.step()

                netG.train()
                netG.zero_grad()
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
        

            print('Train Epoch {}: score {}'.format(epoch +1,score))   

        tools.save_model(netG, 'wgan_g.pth')
        tools.save_model(netD, 'wgan_d.pth')

    else:
        nz = 10
        netG = Generator(nz=nz, ngf=128, nc=1)
        netD = Critic(nc=1, ndf=128)
        
        tools.load_model(netG, 'wgan_g.pth', device)
        tools.load_model(netD, 'wgan_d.pth', device) 

        with torch.no_grad():
            z = torch.randn(2000, nz, 1, 1, device=device)
            samples = (netG(z) + 1) / 2
            score = scorer(samples)

        print(f'The trained WGAN-GP achieves a score of {score:.5f}')

if __name__ == "__main__":
    main()
