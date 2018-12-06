import numpy as np
from PIL import Image

WIDTH = int(16 * 100 * .6)
HEIGHT = int(9 * 100 * .6)


def init_solid(width, height, rgb):
    a = np.array(rgb)
    tiled = np.tile(a, width * height)

    return np.reshape(tiled, (height, width, 3))

def gradient(a, rgb):
    """
    Calculate deltas and add them to all points in a.

    delta = cur_step/(step_count - 1) * (end_color - start_color)
    """
    negated_a = a.copy() * -1
    color_deltas = negated_a.reshape(-1, 3) + rgb
    scale_factors = np.arange(a.shape[0]) / (a.shape[0] - 1)
    scaling = np.repeat(scale_factors, a.shape[1] * a.shape[2])
    scaled_deltas = scaling.reshape(a.shape) * color_deltas.reshape(a.shape)

    return a + scaled_deltas

def save_image(file_name, image_data):
    img = Image.fromarray(np.uint8(image_data), 'RGB')
    img.save(file_name)

def main():
    solid = init_solid(WIDTH, HEIGHT, [0, 255, 0])
    save_image('solid.png', solid)

    g = gradient(solid, [0, 0, 255])
    save_image('gradient.png', g)

if __name__== "__main__":
    main()

