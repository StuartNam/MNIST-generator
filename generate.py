import torch

import matplotlib.pyplot as plt
from model import DiffusionModel
from config import *

def denormalize(x):
    return (x + 1) / 2 * 255

model = DiffusionModel(NUM_TIMESTEPS)
model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))

#fig, (ax1, ax2) = plt.subplots(1, 2)

x0 = model.sample(1)

x0 = denormalize(x0)
x0 = x0.reshape(28, 28)
plt.imshow(x0, cmap = 'gray')
plt.show()
