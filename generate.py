import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import PIL.Image as Image

from model import DiffusionModel
from config import *

def scale_up(x):
    # Scale from [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255

    return x

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiffusionModel(NUM_TIMESTEPS).to(device)

model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location = torch.device(device)))

x0, results = model.sample(1)

x0 = scale_up(x0)
results = [scale_up(result) for result in results]

x0, denoising_process = model.sample(1)

denoising_process = [Image.fromarray(frame.reshape(28, 28).numpy(), mode = 'L') for frame in denoising_process]
denoising_process += [denoising_process[-1] for _ in range(0, 10)]
denoising_process[0].save(RESULT_PATH + 'sample.gif', format = 'GIF', append_images = denoising_process[1:], save_all = True, duration = 100, loop = 0)

# fig = plt.figure()

# def animate(i):
#     plt.imshow(denoising_process[i].cpu().reshape(28, 28), cmap = 'gray')

# ani = animation.FuncAnimation(fig, animate, frames = len(denoising_process), interval = 25)

# plt.show()

# fig, axes = plt.subplots(5, 10)
# for i, axis in enumerate(axes.flat):
#     axis.imshow(results[i].reshape(28, 28).cpu(), cmap = 'gray')

# plt.show()
