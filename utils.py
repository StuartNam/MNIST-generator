import numpy as np
import matplotlib.pyplot as plt

def scale_up(x):
    # Scale from [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255

    return x

def scale_down(x):
    # Scale from [0, 255] to [-1, 1]
    return 2 * x / 255 - 1

def imshow_pairs(imgs1, imgs2, num_columns = 10, cmap = 'gray'):
    num_pairs = imgs1.shape[0]
    pairs = np.concatenate([imgs1, imgs2], dim = 2)
    
    num_rows = num_pairs // num_columns + num_pairs % num_columns == 0
    _, axes = plt.subplots(num_rows, num_columns)

    for i, pair in enumerate(pairs):
        axes[i // num_columns, i % num_columns].imshow(pair, cmap = cmap)
    
    plt.show()

def imshow_imgs(imgs, num_columns = 10, cmap = 'gray'):
    num_imgs = len(imgs)
    num_rows = num_imgs // num_columns + num_imgs % num_columns == 0

    _, axes = plt.subplots(num_rows, num_columns)

    for i, img in enumerate(imgs):
        axes[i // num_columns, i % num_columns].imshow(img, cmap = cmap)
    
    plt.show()