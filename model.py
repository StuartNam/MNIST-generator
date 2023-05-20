import torch
import torch.nn as nn

import math
import random
import numpy as np
from config import *

class DiffusionModel(nn.Module):
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps, dtype = torch.float64)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0, dtype = torch.float64)
        self.sigma = torch.sqrt(self.betas)
        
        self.fc1 = nn.Linear(
            in_features = 784,
            out_features = 1000,
            dtype = torch.float64
        )

        self.fc2 = nn.Linear(
            in_features = 1000,
            out_features = 1000,
            dtype = torch.float64
        )

        self.fc3 = nn.Linear(
            in_features = 1000,
            out_features = 1000,
            dtype = torch.float64
        )

        self.fc4 = nn.Linear(
            in_features = 1000,
            out_features = 784,
            dtype = torch.float64
        )

    def noise_sample(self, x0):
        # x0: (batch_size, 784)
        batch_size = x0.shape[0]
        t = random.randint(1, NUM_TIMESTEPS)
        epsilon = torch.randn(batch_size, 784, dtype = torch.float64)
        noise = (1 - self.alpha_bars[t - 1]) * epsilon
        xt = torch.sqrt(self.alpha_bars[t - 1]) * x0 + noise

        # xt    : (batch_size, 784)
        # noise : (batch_size, 784)
        # t     : int
        return xt, noise, t
    
    def sinusoidal_embedding(self, t):
        embedding_vector = torch.tensor([math.sin(t / 10000 ** (i / 784)) if i % 2 == 0 else math.cos(t / 10000 ** (i / 784)) for i in range(784)], dtype = torch.float64)
        return embedding_vector
    
    def forward(self, time_embedded_xt):
        noise = self.fc1(time_embedded_xt)
        noise = self.fc2(noise)
        noise = self.fc3(noise)
        noise = self.fc4(noise)

        return noise

    def sample(self, num_samples):
        with torch.no_grad():
            xt = torch.randn((num_samples, 784), dtype = torch.float64)
            for t in range(num_samples, 0, -1):
                z = torch.randn((num_samples, 784)) if t > 1 else torch.zeros((num_samples, 784), dtype = torch.float64)
                time_embedded_xt = xt + self.sinusoidal_embedding(t)

                xt = 1 / (torch.sqrt(self.alphas[t - 1])) * (xt - (1 - self.alphas[t - 1]) / (torch.sqrt(1 - self.alpha_bars[t - 1])) * self.forward(time_embedded_xt)) + self.sigma[t - 1] * z
            return xt