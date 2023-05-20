from keras.datasets import mnist
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

from model import DiffusionModel
from config import *

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

class MNISTDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def normalize(x):
    return 2 * x / 255 - 1

train_X = normalize(train_X)

train_dataset = MNISTDataset(torch.from_numpy(train_X.astype(float)))
dataloader = DataLoader(
    dataset = train_dataset, 
    batch_size = BATCH_SIZE, 
    shuffle = True
)
     
# Define model
model = DiffusionModel(NUM_TIMESTEPS)

# Loss function
mse_loss = nn.functional.mse_loss

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr = 0.001
)

# Training loop
num_batches = math.ceil(len(dataloader.dataset) / BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    for batch_no, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # batch: (batch_size, 28, 28)
        batch = batch.reshape(-1, 784)
        xt, true_noise, t = model.noise_sample(batch)
        xt += model.sinusoidal_embedding(t)
        
        predicted_noise = model.forward(xt)

        loss = mse_loss(true_noise, predicted_noise)
        loss.backward()
        
        optimizer.step()

        if batch_no % 100 == 0:
            print("Epoch {}/{}, Batch {}/{}: Loss = {}".format(epoch, NUM_EPOCHS, batch_no, num_batches, loss))

torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)
