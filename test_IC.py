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
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default=10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default=0.01)')
    parser.add_argument('--nz', type=int, default=25,
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
    test_dataset = datasets.MNIST(root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor())
    #test_dataset.data = test_dataset.data[:100]
    #test_dataset.targest = test_dataset.targets[:100]
    test_loader = DataLoader(test_dataset, shuffle=False,
        batch_size=args.batch_size)

    PATH = args.save_dir + '/model_g_02_25_epoch_30.pth'
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model_g = ModelG(args.nz).to(device)
    model_g.load_state_dict(torch.load(PATH))

    #play around with IC

    class Inverse_Classifier(nn.Module):

        def __init__(self, gen_net, num_classes=10, loss_function_inv=torch.nn.MSELoss()):

            super(Inverse_Classifier, self).__init__()
            self.num_classes = num_classes
            self.loss_function_inv = loss_function_inv
            self.gen_net = gen_net

        def inverse_classification(self, target, plot_result=False, plot_iter=False, verbose=False, lr = 0.01, num_iter=50, alpha=1, update_after_iter=10):

            def sgd_step(input_vector, gradient_vector, eta, V_t_1=0, momentum=0.9):

                V_t = momentum * V_t_1 + (1 - momentum) * gradient_vector
                output_vector = input_vector - eta * V_t
                return(output_vector, V_t)

            def adam_step(input_vector, gradient_vector, m_t1=0, v_t1=0, beta1=0.9, beta2=0.99, eta=0.0001, epsilon=1e-8):

                # get moment terms
                m_t = torch.mul(m_t1, beta1) + torch.mul((1 - beta1), gradient_vector)
                v_t = v_t1 * beta2 + (1 - beta2) * torch.mul(gradient_vector, gradient_vector)

                epsilon = epsilon * torch.zeros(gradient_vector.size(), device=device)
                dx = (eta * m_t) / (torch.sqrt(v_t) + epsilon)
                output_vector = input_vector - dx

                return (output_vector, m_t, v_t)

            self.input = torch.Tensor([1/self.num_classes]*self.num_classes).repeat(target.size(0), 1).view(target.size(0), 10).to(device)
            self.noise = torch.FloatTensor(target.size(0), (args.nz)).to(device)
            self.noise.resize_(target.size(0), args.nz).normal_(0,1)

            target = target.view(-1, 28,28)

            m_t1 = torch.zeros(target.size(0), self.num_classes, device=device)
            m_t1_ = torch.zeros(target.size(0), args.nz, device=device)
            V_t_1 = torch.zeros(target.size(0), self.num_classes, device=device)
            V_t_1_ = torch.zeros(target.size(0), args.nz, device=device)

            for i in range(num_iter):

                self.input = self.input.clamp(min=0, max=1)
                self.input /= self.input.sum(dim=1).view(-1,1)
                self.input = torch.autograd.Variable(self.input, requires_grad=True)
                self.noise = torch.autograd.Variable(self.noise, requires_grad=True)

                output = self.gen_net(self.noise, self.input).view(target.size(0), 28,28)
                loss = self.loss_function_inv(output, target)
                if verbose:
                    print("self.input: ", self.input)
                    print('loss: ', loss.item())

                #propagate the loss back
                loss.backward()

                input_gradient = self.input.grad
                noise_gradient = self.noise.grad

                #self.input, V_t_1 = sgd_step(input_vector=self.input, gradient_vector=input_gradient, eta=lr, V_t_1=V_t_1)
                #self.noise, V_t_1_ = sgd_step(input_vector=self.noise, gradient_vector=noise_gradient, eta=lr, V_t_1=V_t_1_)
                self.input, m_t1, V_t_1 = adam_step(input_vector=self.input, gradient_vector=input_gradient, eta=lr, m_t1=m_t1, v_t1=V_t_1)
                if i > update_after_iter:
                    self.noise, m_t1_, V_t_1_ = adam_step(input_vector=self.noise, gradient_vector=noise_gradient, eta=alpha*lr, m_t1=m_t1_, v_t1=V_t_1_)
                if plot_iter:
                    img = output.view(28, 28).detach().cpu()
                    plt.axis('off')
                    plt.imshow(img)
                    plt.show()

            if plot_result:
                fig, axs = plt.subplots(nrows=target.size(0), ncols=2, sharex=False, figsize=(10, 7 * target.size(0)))

                for i in range(target.size(0)):
                    axs[i][0].imshow(target[i].view(28,28).detach().cpu().numpy())
                    axs[i][0].set_title('target')
                    axs[i][0].axis('off')
                    axs[i][1].imshow(output[i].view(28,28).detach().cpu().numpy())
                    axs[i][1].set_title('prediction')
                    axs[i][1].axis('off')

                plt.show()

            self.input = self.input.clamp(min=0, max=1)
            self.input /= self.input.sum()

            _, pred = self.input.max(1)
            return(pred)


    IC = Inverse_Classifier(gen_net=model_g)

    true_classes = []
    pred_classes = []

    for x, y in test_loader:
        x, y = x.float().view(x.size(0), 28, 28).to(device), y.to(device)
        pred = IC.inverse_classification(target=x, verbose=False, plot_result=True, num_iter=200, lr = 0.05, alpha=1, update_after_iter=-1)
        true_classes.append(y)
        pred_classes.append(pred)
        #print("pred, true: ", pred, y)

    true_classes = torch.cat(true_classes).cpu()
    pred_classes = torch.cat(pred_classes).cpu()
    acc = (true_classes == pred_classes).float().mean()
    print("accuracy: ", acc)
