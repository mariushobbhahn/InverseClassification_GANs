""" Conditional DCGAN for MNIST images generations.
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import os
import argparse
import numpy as np
import torch
from torch import nn, optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torch.distributions as tdist

import matplotlib.pyplot as plt

from model import ModelD, ModelG

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DCGAN')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size (default=10)')
    parser.add_argument('--nz', type=int, default=50,
                        help='Number of dimensions for input noise.')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Enable cuda')
    parser.add_argument('--save_dir', type=str, default='models',
            help='Path to save the trained models.')
    parser.add_argument('--samples_dir', type=str, default='samples',
            help='Path to save the output samples.')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.samples_dir):
        os.mkdir(args.samples_dir)

    INPUT_SIZE = 784
    SAMPLE_SIZE = 80
    NUM_LABELS = 10
    train_dataset = datasets.MNIST(root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True,
        batch_size=args.batch_size)

    PATH = args.save_dir + '/model_g_0.2_50_epoch_24.pth'
    model_g = ModelG(args.nz)
    model_g.load_state_dict(torch.load(PATH))
    z_size = 10
    for row in range(10):
        noise = torch.FloatTensor(args.batch_size, (args.nz))
        noise.resize_(args.batch_size, args.nz).normal_(0,1)
        one_hot = torch.eye(10)
        test = model_g(noise, one_hot)
        for col, img in enumerate(test):
            img = img.view(28, 28).detach().cpu()
            plt.subplot(10, 10, row*10 + col + 1)
            plt.axis('off')
            plt.imshow(img)

    plt.show()
