import torch

import matplotlib.pyplot as plt
from model import DiffusionModel
from config import *

def scale_up(x):
    # Scale from [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255

    return x

model = DiffusionModel(NUM_TIMESTEPS)
model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))

x0, results = model.sample(1)

x0 = scale_up(x0)
results = [scale_up(result) for result in results]

fig, axes = plt.subplots(5, 10)
for i, axis in enumerate(axes.flat):
    axis.imshow(results[i].reshape(28, 28), cmap = 'gray')

plt.show()
