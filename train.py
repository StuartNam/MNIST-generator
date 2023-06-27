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
model = DiffusionModel(NUM_TIMESTEPS)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.noise_estimator.parameters(),
    lr = LEARNING_RATE
)

# Training loop
num_batches = math.ceil(len(dataloader.dataset) / BATCH_SIZE)

model.noise_estimator.train()

for epoch in range(NUM_EPOCHS):
    for batch_no, x0 in enumerate(dataloader):
        # t = random.randint(1, NUM_TIMESTEPS)
        t = 50
        xt, e = model.sample_xt_from_x0(x0, t)

        xt_with_t_embedded = model.sinusoidal_embed(xt, t)
      
        noise_predicted = model.noise_estimator(xt_with_t_embedded)

        loss = loss_fn(e, noise_predicted)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_no == 0:
            fig, axes = plt.subplots(1, 2)
            fig.canvas.manager.set_window_title(t)
            
            axes[0].imshow(e[0].reshape(28, 28), cmap = "gray")
            axes[1].imshow(noise_predicted[0].detach().numpy().reshape(28, 28), cmap = "gray")
            
            plt.show()

            model.noise_estimator.eval()
            with torch.no_grad():
                test_x0 = model.sample(10)
                test_x0 = scale_up(test_x0)

                _, axes = plt.subplots(1, 10)
                
                for i in range(10):
                    axes[i].imshow(test_x0[i].reshape(28, 28), cmap = "gray")
                
                plt.show()
                
            model.noise_estimator.train()

        if batch_no % 100 == 0:
            print("Epoch {}/{}, Batch {}/{}: loss = {}".format(epoch + 1, NUM_EPOCHS, batch_no + 1, num_batches, loss))

    if epoch != 0 and epoch % 100 == 0:
        choice = input("End training? (Y/N): ")
        if choice == "Y":
            break
        
torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)
