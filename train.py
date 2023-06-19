from keras.datasets import mnist
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

from model import DiffusionModel
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device =", device)
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

class MNISTDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def scale_down(x):
    # Scale pixel values down to [-1, 1]
    return 2 * x / 255 - 1

def scale_up(x):
    # Scale from [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255

    return x

train_X = scale_down(train_X)

train_dataset = MNISTDataset(torch.from_numpy(train_X).to(torch.float32).to(device))
dataloader = DataLoader(
    dataset = train_dataset, 
    batch_size = BATCH_SIZE, 
    shuffle = True
)
     
# Define model
model = DiffusionModel(NUM_TIMESTEPS).to(device)

# Loss function
mse_loss = nn.functional.mse_loss

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr = LEARNING_RATE
)

# Training loop
num_batches = math.ceil(len(dataloader.dataset) / BATCH_SIZE)

model.train()
for epoch in range(NUM_EPOCHS):
    for batch_no, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # batch: (batch_size, 28, 28)
        batch = batch.reshape(-1, 1, 784)
        xt, true_noise, t = model.noise_sample(batch)
        xt += model.sinusoidal_embedding(t)
        
        xt = xt.reshape(-1, 1, 28, 28)
        true_noise = true_noise.reshape(-1, 1, 28, 28)
        
        predicted_noise = model.forward(xt)

        loss = mse_loss(true_noise, predicted_noise)
        loss.backward()
        
        optimizer.step()

        if batch_no % 100 == 0:
            print("Epoch {}/{}, Batch {}/{}: Loss = {}".format(epoch, NUM_EPOCHS, batch_no, num_batches, loss))

    # if epoch % 10 == 0:
    #     x0, denoising_process = model.sample(1)

    #     x0 = scale_up(x0)
    #     denoising_process = [scale_up(result) for result in denoising_process]

    #     fig = plt.figure()

    #     def animate(i):
    #         plt.imshow(denoising_process[i].cpu().reshape(28, 28), cmap = 'gray')

    #     ani = animation.FuncAnimation(fig, animate, frames = len(denoising_process), interval = 50)

    #     plt.show()
    #     # fig, ax = plt.subplots()

    #     # plot = ax.imshow(torch.zeros_like(denoising_process[0].reshape(28, 28)).cpu(), cmap = 'gray')

    #     # def update(frame):
    #     #     plot.set_data(denoising_process[frame].cpu().reshape(28, 28))
    #     #     return plot,

    #     # animation = FuncAnimation(fig, update, frames = len(denoising_process), interval = 50)

    #     # plt.show()

    #     # fig, axes = plt.subplots(10, 10)
    #     # for i, axis in enumerate(axes.flat):
    #     #     axis.imshow(denoising_process[i].reshape(28, 28), cmap = 'gray')

    #     # plt.show()

    #     model.train()

torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)
