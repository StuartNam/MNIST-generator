def scale_up(x):
    # Scale from [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255

    return x

def scale_down(x):
    # Scale pixel values down to [-1, 1]
    return 2 * x / 255 - 1