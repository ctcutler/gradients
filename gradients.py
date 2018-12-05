import numpy as np
from PIL import Image

WIDTH = int(16 * 100 * .6)
HEIGHT = int(9 * 100 * .6)


def init_solid(width, height, rgb):
    a = np.array(rgb)
    tiled = np.tile(a, width * height)
    return np.reshape(tiled, (height, width, 3))

def init_gradient(width, height, rgb1, rgb2, gradient_func):
    a = np.empty((height, width, 3))
    # FIXME: refactor based on gradient() approach?
    for i in range(3):
        gradient = gradient_func(rgb1[i], rgb2[i], height)
        repeated = np.repeat(gradient, width)
        shaped = np.reshape(repeated, (height, width))
        a[:,:,i] = shaped
    return a

def gradient(a, rgb):
    """
    for now:
    - direction is always down
    - rate of change is always linear

    delta = cur step/total steps * (end - start)

    create a delta array and add it to the existing one. . . to do that I would need to:
    - make a copy of the array
        - multiply everything by -1
        - add the end value to everything
    - make a new array with the same size/shape
        - fill every row with its index/(height-1)
    - multiply these arrays together and add result to the original array
    """
    # FIXME: give this a good scrubbing
    negated = np.copy(a) * -1
    color_triples = negated.reshape(-1, 3)
    deltas_flat = color_triples + rgb
    deltas = np.reshape(deltas_flat, a.shape)
    scale_factors = np.arange(a.shape[0]) / (a.shape[0] - 1)
    scale_factors_flat = np.repeat(scale_factors, a.shape[1] * a.shape[2])
    scale_factors_reshaped = np.reshape(scale_factors_flat, a.shape)
    return a + (scale_factors_reshaped * deltas)

def save_image(file_name, image_data):
    img = Image.fromarray(np.uint8(image_data), 'RGB')
    img.save(file_name)

solid = init_solid(WIDTH, HEIGHT, [0, 255, 0])
save_image('solid.png', solid)

g = gradient(solid, [0, 0, 255])
save_image('gradient.png', g)
