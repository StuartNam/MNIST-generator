import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import math
import random
import numpy as np
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

#======================================================================================#
#   MODEL DEFINITION
#   - Define model in this file
#======================================================================================#

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.down_block_1 = ConvBlock(
            num_convlayers = 2,
            num_in_channels = 1,
            channels_scale = 4
        )

        self.down_block_2 = ConvBlock(
            num_convlayers = 2,
            num_in_channels = 16,
            channels_scale = 4
        )

        self.bottle_neck = nn.Conv2d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = 3,
            padding = 1,
            dtype = DTYPE
        )

        self.up_block_1 = ConvTranposeBlock(
            num_convlayers = 1,
            num_in_channels = 512,
            num_out_channels = 16
        )

        self.up_block_2 = ConvTranposeBlock(
            num_convlayers = 1,
            num_in_channels = 32,
            num_out_channels = 1,
            out = True
        )

    def forward(self, x):
        # y_down_1: (N, 16, 14, 14)
        y_down_1 = self.down_block_1(x)
        y_down_1_clone = y_down_1.clone()
        y_down_1_clone.retain_grad()

        # y_down_2: (N, 256, 7, 7)
        y_down_2 = self.down_block_2(y_down_1)
        y_down_2_clone = y_down_2.clone()
        y_down_2_clone.retain_grad()

        # y_down_3: (N, 256, 7, 7)
        y_down_3 = self.bottle_neck(y_down_2)

        # y_down_3_plus_2_clone: (N, 512, 7, 7)
        y_down_3_plus_2_clone = torch.cat([y_down_2_clone, y_down_3], dim = 1)

        # y_up_1: (N, 16, 14, 14)
        y_up_1 = self.up_block_1(y_down_3_plus_2_clone)

        # y_down_1_plus_1_clone: (N, 32, 14, 14)
        y_up_1_plus_1_clone = torch.cat([y_down_1_clone, y_up_1], dim = 1)

        # y_up_2: (N, 1, 28, 28)
        y_up_2 = self.up_block_2(y_up_1_plus_1_clone)

        return y_up_2
    
class DiffusionModel():
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(
            start = 0.0001, 
            end = 0.02, 
            steps = num_timesteps,
            dtype = DTYPE,
            device = device
        )

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(
            self.alphas, 
            dim = 0, 
            dtype = DTYPE
        )

        self.sigma = torch.sqrt(self.betas)

        self.noise_estimator = UNet().to(device)

    def sinusoidal_embed(self, x, t):
        """
            Embed image xt with t to input into Noise Estimator
            In:
            - x: (N, 1, 784)
            - t: int
            Out:
            - x_with_t_embedded: (N, 1, 784)
        """

        t_encoded = torch.tensor(
            data = [math.sin(t / self.num_timesteps ** (i / 784 / 2)) if i % 2 == 0 else math.cos(t / self.num_timesteps ** (i / 784 / 2)) for i in range(784)], 
            dtype = torch.float32,
            device = device
        )

        t_encoded = t_encoded.reshape(1, 784)

        x_with_t_embedded = x + t_encoded

        return x_with_t_embedded
        
    def noise_sample(self, x0):
        """
            In:
            - x0: (N, 1, 784)
            Out:
            - xt: (N, 1, 784)
            - noise: (N, 1, 784)
            - t: int
        """

        t = random.randint(1, self.num_timesteps)
        
        # epsilon: (N, 1, 784)
        epsilon = torch.randn_like(x0)

        # noise = (1 - self.alpha_bars[t - 1]) ** 0.5 * epsilon
        xt = self.alpha_bars[t - 1] ** 0.5 * x0 + (1 - self.alpha_bars[t - 1]) ** 0.5 * epsilon

        return xt, epsilon, t
        
    def sample(self):
        self.noise_esimator.eval()
        with torch.no_grad():
            xt = torch.randn(
                size = (1, 1, 784),
                dtype = DTYPE,
                device = device
            )

            for t in range(self.num_timesteps, 0, -1):
                epsilon = torch.rand_like(xt)
                noise_predicted = self.noise_esimator(self.sinusoidal_embed(xt, t))

                xt = 1 / self.alphas[t - 1] ** 0.5 * (xt - (1 - self.alphas[t - 1]) / (1 - self.alpha_bars[t - 1]) ** 0.5 * noise_predicted) + self.sigma[t - 1] * epsilon

            return xt

class ConvBlock(nn.Module):
    def __init__(self, num_convlayers, num_in_channels, channels_scale):
        super().__init__()

        self.layers = nn.Sequential()

        for i in range(num_convlayers):
            self.layers.add_module(
                name = "Conv{}".format(i),
                module = nn.Conv2d(
                    in_channels = num_in_channels * (channels_scale ** i),
                    out_channels = num_in_channels * (channels_scale ** (i + 1)),
                    kernel_size = 3,
                    padding = 1,
                    dtype = DTYPE
                )
            )
        
        self.layers.add_module(
            name = "BN",
            module = nn.BatchNorm2d(
                num_features = num_in_channels * (channels_scale ** num_convlayers),
                dtype = DTYPE
            )
        )

        self.layers.add_module(
            name = "MP",
            module = nn.MaxPool2d(
                kernel_size = 2,
                stride = 2
            )
        )

        self.layers.add_module(
            name = "ReLU",
            module = nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class ConvTranposeBlock(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, num_convlayers, out = False):
        super().__init__()

        self.layers = nn.Sequential()

        for i in range(num_convlayers):
            self.layers.add_module(
                name = "ConvTranspose{}".format(i),
                module = nn.ConvTranspose2d(
                    in_channels = num_in_channels,
                    out_channels = num_in_channels,
                    kernel_size = 2,
                    stride = 2,
                    dtype = DTYPE
                )
            )
        
        self.layers.add_module(
            name = "Conv",
            module = nn.Conv2d(
                in_channels = num_in_channels,
                out_channels = num_out_channels,
                kernel_size = 1,
                dtype = DTYPE
            )
        )

        self.layers.add_module(
            name = "BN",
            module = nn.BatchNorm2d(
                num_features = num_out_channels,
                dtype = DTYPE
            )
        )

        if not out:
            self.layers.add_module(
                name = "ReLU",
                module = nn.ReLU()
            )
        
    def forward(self, x):
        return self.layers(x)
    
class Reshape(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()

        self.new_shape = (C, H, W)

    def forward(self, x):
        return x.reshape(-1, self.new_shape[0], self.new_shape[1], self.new_shape[2])