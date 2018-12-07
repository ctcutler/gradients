import numpy as np
from PIL import Image

WIDTH = int(16 * 100 * .6)
HEIGHT = int(9 * 100 * .6)


def init_solid(width, height, rgb):
    a = np.array(rgb)
    tiled = np.tile(a, width * height)

    return np.reshape(tiled, (height, width, 3))

def gradient(a, rgb, scale_func):
    """
    Calculate deltas and add them to all points in a.

    delta = scale_func(cur_step, step_count) * (new_color - existing_color)
    """

    (height, width, depth) = a.shape
    negated_a = a.copy() * -1
    color_deltas = negated_a.reshape(-1, 3) + rgb
    scale_factors = np.array([scale_func(x, height) for x in range(height)])
    scaling = np.repeat(scale_factors, width * depth)
    scaled_deltas = scaling.reshape(a.shape) * color_deltas.reshape(a.shape)

    return a + scaled_deltas

def noise(a, amount):
    r = (np.random.rand(*a.shape) * amount * 2) - amount
    noisy = r + a
    upper_limit = np.repeat(255, a.size).reshape(a.shape)
    lower_limit = np.repeat(0, a.size).reshape(a.shape)
    return np.minimum(upper_limit, np.maximum(lower_limit, noisy))

def save_image(file_name, image_data):
    img = Image.fromarray(np.uint8(image_data), 'RGB')
    img.save(file_name)

def color_scale(cur, total):
    """
    Goal is to make a certain proportion solid and do the gradient
    only after/above that proportion.
    """
    proportion = .85

    if cur >= total * proportion:
        n = (cur - (total * proportion)) * \
            (total/(total-(total*proportion)))
        return float(n**2) / ((total - 1)**2)
    else:
        return 0

def main():
    blue = (0, 65, 171)
    yellow = (255, 232, 0)

    solid = init_solid(WIDTH, HEIGHT, blue)
    save_image('solid.png', solid)


    (top, bottom) = np.split(solid, 2)
    gradient_top = gradient(top, yellow, color_scale)
    gradient_bottom = gradient(bottom, yellow, color_scale)
    gradient_bottom_flipped = np.flipud(gradient_bottom)
    g = np.concatenate((gradient_top, gradient_bottom_flipped))
    g_noise = noise(g, 5)
    save_image('gradient.png', g_noise)

if __name__== "__main__":
    main()

