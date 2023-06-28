from keras.datasets import mnist

import numpy as np

import math
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import random

from model import DiffusionModel
from config import *
from utils import *
from dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device)

# Load dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = np.concatenate([train_x, test_x], axis = 0)
train_x = torch.tensor(train_x, dtype = DTYPE)
train_x = scale_down(train_x)
train_x = train_x.reshape(-1, 1, 28, 28)
train_x = train_x.to(device)

train_dataset = MNISTDataset(train_x)
dataloader = DataLoader(
    dataset = train_dataset, 
    batch_size = BATCH_SIZE, 
    shuffle = True
)
     
# Define model
model = DiffusionModel(NUM_TIMESTEPS, DTYPE)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer1 = torch.optim.Adam(
    model.noise_estimator1.parameters(),
    lr = LEARNING_RATE
)

optimizer2 = torch.optim.Adam(
    model.noise_estimator2.parameters(),
    lr = LEARNING_RATE
)

optimizer3 = torch.optim.Adam(
    model.noise_estimator3.parameters(),
    lr = LEARNING_RATE
)

optimizer4 = torch.optim.Adam(
    model.noise_estimator4.parameters(),
    lr = LEARNING_RATE
)

# Training loop
num_batches = math.ceil(len(dataloader.dataset) / BATCH_SIZE)

model.train()
for epoch_no in range(NUM_EPOCHS):
    if (epoch_no + 1) % 100 == 0:
        choice = input("End training? (Y/N): ")
        if choice == "Y":
            break

    for batch_no, x0 in enumerate(dataloader):
        # Report
        if (epoch_no + 1) % 5 == 0 and batch_no == 0:
            model.eval()
            with torch.no_grad():
                test_x0 = x0[0]
                _, axes = plt.subplots(10, 10)
                for t in range(1, NUM_TIMESTEPS):
                    test_xt, e = model.sample_xt_from_x0(test_x0, t)
                    xt_and_e = torch.cat([xt, e], dim = 3)

                    axes[t // 10, t % 10].imshow(xt_and_e.reshape(28, 56).cpu(), cmap = 'gray')
                
                plt.show()

                test_x0, test_xT_0 = model.sample(1)

            model.eval()
            with torch.no_grad():
                test_x0 = model.sample(10)
                test_x0 = scale_up(test_x0)

                test_x0 = scale_up(test_x0).reshape(28, 28)
                test_xT_0 = [scale_up(xt).reshape(28, 28) for xt in test_xT_0]

                imshow_imgs(test_xT_0)

                plt.imshow(test_x0, cmap = 'gray')
                plt.show()
                
            model.train()

        t = random.randint(1, NUM_TIMESTEPS)

        xt, e = model.sample_xt_from_x0(x0, t)

        xt_with_t_embedded = model.sinusoidal_embed(xt, t)
      
        noise_predicted = model.predict_noise(xt_with_t_embedded, t)

        loss = loss_fn(e, noise_predicted)
       
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        loss.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        if batch_no % 100 == 0:
            print("Epoch {}/{}, Batch {}/{}: loss = {}, at timestep {}".format(epoch_no + 1, NUM_EPOCHS, batch_no + 1, num_batches, loss, t))
        
model.save()
