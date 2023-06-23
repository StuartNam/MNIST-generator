import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import math
import random
import numpy as np
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps, dtype = torch.float32).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0, dtype = torch.float32)
        self.sigma = torch.sqrt(self.betas)
        
        self.relu = nn.ReLU()

        # In: (N, 1, 28, 28)  Out: (N, 64, 28, 28)
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            dtype = torch.float32
        )

        self.bn1 = nn.BatchNorm2d(
            num_features = 64,
            dtype = torch.float32
        )

        # In: (N, 64, 28, 28)  Out: (N, 64, 14, 14)
        self.maxpool1 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        # In: (N, 64, 14, 14)  Out: (N, 256, 14, 14)
        self.conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            dtype = torch.float32
        )
        
        self.bn2 = nn.BatchNorm2d(
            num_features = 256,
            dtype = torch.float32
        )

        # In: (N, 256, 14, 14)  Out: (N, 256, 7, 7)
        self.maxpool2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        # In: (N, 256, 7, 7)  Out: (N, 256, 14, 14)
        self.upconv1 = nn.ConvTranspose2d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = 2,
            stride = 2,
            dtype = torch.float32
        )
        
        self.bn3 = nn.BatchNorm2d(
            num_features = 256,
            dtype = torch.float32
        )

        # In: (N, 512, 14, 14)  Out: (N, 64, 14, 14)
        self.conv3 = nn.Conv2d(
            in_channels = 512,
            out_channels = 64,
            kernel_size = 1,
            dtype = torch.float32
        )

        self.bn4 = nn.BatchNorm2d(
            num_features = 64,
            dtype = torch.float32
        )

        # In: (N, 64, 14, 14)  Out: (N, 64, 28, 28)
        self.upconv2 = nn.ConvTranspose2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 2,
            stride = 2,
            dtype = torch.float32
        )

        self.bn5 = nn.BatchNorm2d(
            num_features = 64,
            dtype = torch.float32
        )

        # In: (N, 128, 28, 28)  Out: (N, 1, 28, 28)
        self.conv4 = nn.Conv2d(
            in_channels = 128,
            out_channels = 1,
            kernel_size = 1,
            dtype = torch.float32
        )

        
    def noise_sample(self, x0):
        self.eval()
        with torch.no_grad():
            # x0: (N, 1, 784)
            batch_size = x0.shape[0]
            
            # t: int
            t = random.randint(1, NUM_TIMESTEPS)

            # epsilon: (N, 1, 784)
            epsilon = torch.randn(batch_size, 1, 784, dtype = torch.float32).to(device)

            # true_noise: (N, 1, 784)
            noise = (1 - self.alpha_bars[t - 1]) * epsilon
            xt = torch.sqrt(self.alpha_bars[t - 1]) * x0 + noise

        # xt        : (N, 1, 784)
        # noise     : (N, 1, 784)
        # t         : int
        return xt, noise, t
    
    def sinusoidal_embedding(self, t):
        self.eval()
        with torch.no_grad():
            # embedding_vector: (784, )
            embedding_vector = torch.tensor([math.sin(t / 10000 ** (i / 784)) if i % 2 == 0 else math.cos(t / 10000 ** (i / 784)) for i in range(784)], dtype = torch.float32).to(device)

            # embedding_vector: (1, 784)
            embedding_vector = embedding_vector.reshape(1, 784)

        # embedding_vector: (1, 784)
        return embedding_vector
    
    def forward(self, time_embedded_xt):
        # time_embedded_xt: (N, 1, 28, 28)

        noise = self.conv1(time_embedded_xt)
        noise = self.bn1(noise)
        noise = self.relu(noise)
        tmp1 = noise.clone()
        noise = self.maxpool1(noise)

        noise = self.conv2(noise)
        noise = self.bn2(noise)
        noise = self.relu(noise)
        tmp2 = noise.clone()
        noise = self.maxpool2(noise)

        noise = self.upconv1(noise)
        noise = self.bn3(noise)
        noise = self.relu(noise)
        noise = torch.concatenate((tmp2, noise), dim = 1)

        noise = self.conv3(noise)
        noise = self.bn4(noise)
        noise = self.relu(noise)

        noise = self.upconv2(noise)
        noise = self.bn5(noise)
        noise = self.relu(noise)
        noise = torch.concatenate((tmp1, noise), dim = 1)

        noise = self.conv4(noise)

        # noise: (N, 1, 28, 28)
        return noise

    def sample(self, num_samples):
        with torch.no_grad():
            self.eval()

            # xt: (N, 1, 784)
            xt = torch.randn((num_samples, 1, 784), dtype = torch.float32).to(device)
            denoising_process = []
            noises_predicted = []

            for t in range(NUM_TIMESTEPS, 0, -1):
                # xt: (N, 1, 784)
                xt = xt.reshape(-1, 1, 784)

                # z: (N, 1, 784)
                z = torch.randn((num_samples, 1, 784)).to(device) if t > 1 else torch.zeros((num_samples, 1, 784), dtype = torch.float32).to(device)

                # time_embedded_xt: (1, 784)
                time_embedded_xt = xt + self.sinusoidal_embedding(t)

                z = z.reshape(-1, 1, 28, 28)
                xt = xt.reshape(-1, 1, 28, 28)
                time_embedded_xt = time_embedded_xt.reshape(-1, 1, 28, 28)

                noise_predicted = self.forward(time_embedded_xt)
                noises_predicted.append(torch.norm(noise_predicted.clone()))

                xt = 1 / (torch.sqrt(self.alphas[t - 1])) * (xt - (1 - self.alphas[t - 1]) / (torch.sqrt(1 - self.alpha_bars[t - 1])) * noise_predicted) + self.sigma[t - 1] * z
                denoising_process.append(xt.clone())

            #print(len(predicted_noises))
            plt.plot(noises_predicted)
            plt.show()

            return xt, denoising_process