import numpy as np
from PIL import Image

WIDTH = int(16 * 100 * .6)
HEIGHT = int(9 * 100 * .6)


def init_solid(width, height, rgb):
    a = np.array(rgb, dtype=np.uint8)
    tiled = np.tile(a, width * height)
    return np.reshape(tiled, (height, width, 3))

def init_gradient(width, height, rgb1, rgb2):
    a = np.empty((height, width, 3))
    for i in range(3):
        linear = np.linspace(rgb1[i], rgb2[i], height)
        repeated = np.repeat(linear, width)
        shaped = np.reshape(repeated, (height, width))
        a[:,:,i] = shaped
    return np.uint8(a)

def save_image(file_name, image_data):
    img = Image.fromarray(image_data, 'RGB')
    img.save(file_name)

solid = init_solid(WIDTH, HEIGHT, [0, 255, 0])
save_image('solid.png', solid)

gradient = init_gradient(WIDTH, HEIGHT, [0, 255, 0], [0, 0, 255])
save_image('gradient.png', gradient)
