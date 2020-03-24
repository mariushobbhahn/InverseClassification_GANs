""" Conditional DCGAN for MNIST images generations.
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import normal

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms

from model import ModelD, ModelG


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DCGAN')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default=128)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default=0.01)')
    parser.add_argument('--epochs', type=int, default=61,
                        help='Number of training epochs.')
    parser.add_argument('--nz', type=int, default=50,
                        help='Number of dimensions for input noise.')
    parser.add_argument('--noise_size_input', type=float, default=0.2)
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Enable cuda')
    parser.add_argument('--save_every', type=int, default=3,
                        help='After how many epochs to save the model.')
    parser.add_argument('--print_every', type=int, default=50,
            help='After how many epochs to print loss and save output samples.')
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

    model_d = ModelD()
    model_g = ModelG(args.nz, num_classes=10)
    criterion = nn.BCELoss()
    input = torch.FloatTensor(args.batch_size, INPUT_SIZE)
    noise = torch.FloatTensor(args.batch_size, (args.nz))

    fixed_noise = torch.FloatTensor(SAMPLE_SIZE, args.nz).normal_(0,1)
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS)
    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i*(SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

    label = torch.FloatTensor(args.batch_size)
    one_hot_labels = torch.FloatTensor(args.batch_size, 10).cuda()
    if args.cuda:
        model_d.cuda()
        model_g.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        one_hot_labels = one_hot_labels.cuda()
        fixed_labels = fixed_labels.cuda()

    optim_d = optim.SGD(model_d.parameters(), lr=args.lr)
    optim_g = optim.SGD(model_g.parameters(), lr=args.lr)
    fixed_noise = Variable(fixed_noise)
    fixed_labels = Variable(fixed_labels)

    real_label = 1
    fake_label = 0

    m = normal.Normal(0.0, args.noise_size_input)

    for epoch_idx in range(args.epochs):
        model_d.train()
        model_g.train()


        d_loss = 0.0
        g_loss = 0.0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)
            train_x = train_x.view(-1, INPUT_SIZE)
            if args.cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            input.resize_as_(train_x).copy_(train_x)
            label.resize_(batch_size).fill_(real_label)
            one_hot_labels.resize_(batch_size, NUM_LABELS).zero_()
            one_hot_labels.scatter_(1, train_y.view(batch_size,1), 1)
            inputv = Variable(input)
            labelv = Variable(label)

            output = model_d(inputv, Variable(one_hot_labels))
            optim_d.zero_grad()
            errD_real = criterion(output, labelv)
            errD_real.backward()
            realD_mean = output.data.cpu().mean()

            one_hot_labels.zero_()
            rand_y = torch.randint(0, NUM_LABELS, size=(batch_size,1), device='cuda')
            one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1)
            noise.resize_(batch_size, args.nz).normal_(0,1)
            label.resize_(batch_size).fill_(fake_label)
            noisev = Variable(noise)
            labelv = Variable(label)
            onehotv = Variable(one_hot_labels)
            g_out = model_g(noisev, onehotv)
            output = model_d(g_out, onehotv)
            errD_fake = criterion(output, labelv)
            fakeD_mean = output.data.cpu().mean()
            errD = errD_real + errD_fake
            errD_fake.backward()
            optim_d.step()

            # train the G
            noise.normal_(0,1)
            one_hot_labels.zero_().cuda()
            rand_y = torch.randint(0, NUM_LABELS, size=(batch_size,1), device='cuda')
            one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1)
            label.resize_(batch_size).fill_(real_label)
            one_hot_labels += m.sample((train_x.size(0), model_g.num_classes)).cuda()
            one_hot_labels = one_hot_labels.clamp(min=0, max=1)
            onehotv = Variable(one_hot_labels)
            noisev = Variable(noise)
            labelv = Variable(label)
            g_out = model_g(noisev, onehotv)
            output = model_d(g_out, onehotv)
            errG = criterion(output, labelv)
            optim_g.zero_grad()
            errG.backward()
            optim_g.step()

            d_loss += errD.item()
            g_loss += errG.item()
            if batch_idx % args.print_every == 0:
                print(
                "\t{} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
                    format(epoch_idx, batch_idx, len(train_loader), fakeD_mean,
                        realD_mean))

                g_out = model_g(fixed_noise, fixed_labels).data.view(
                    SAMPLE_SIZE, 1, 28,28).cpu()
                save_image(g_out,
                    '{}/{}_{}.png'.format(
                        args.samples_dir, epoch_idx, batch_idx))


        print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
            d_loss, g_loss))
        if epoch_idx % args.save_every == 0:
            D_PATH = '{}/model_d_{}_{}_epoch_{}.pth'.format(args.save_dir, args.noise_size_input, args.nz, epoch_idx)
            G_PATH = '{}/model_g_{}_{}_epoch_{}.pth'.format(args.save_dir, args.noise_size_input, args.nz, epoch_idx)
            print("saving G at: {} and D at: {}".format(G_PATH, D_PATH))
            torch.save(model_d.state_dict(), D_PATH)
            torch.save(model_g.state_dict(), G_PATH)
