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
(train_x, train_y), (test_x, test_y) = mnist.load_data()

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

train_x = scale_down(train_x)

train_dataset = MNISTDataset(torch.from_numpy(train_x).to(DTYPE).to(device))
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
optimizer = optim.Adam(
    model.noise_estimator.parameters(),
    lr = LEARNING_RATE
)

# Training loop
num_batches = math.ceil(len(dataloader.dataset) / BATCH_SIZE)

model.noise_estimator.train()

for epoch in range(NUM_EPOCHS):
    for batch_no, x0 in enumerate(dataloader):
        # x0: (N, 1, 784)
        x0 = x0.reshape(-1, 1, 784)
        xt, epsilon, t = model.noise_sample(x0)
        
        xt_with_t_embedded = model.sinusoidal_embed(xt, t)

        xt_with_t_embedded = xt_with_t_embedded.reshape(-1, 1, 28, 28)
        epsilon = epsilon.reshape(-1, 1, 28, 28)
        
        noise_predicted = model.noise_estimator(xt_with_t_embedded)

        loss = loss_fn(epsilon, noise_predicted)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_no % 100 == 0:
            print("Epoch {}/{}, Batch {}/{}: Loss = {}".format(epoch + 1, NUM_EPOCHS, batch_no + 1, num_batches, loss))

    if epoch != 0 and epoch % 100 == 0:
        choice = input("End training? (Y/N): ")
        if choice == "Y":
            break
        
torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)
