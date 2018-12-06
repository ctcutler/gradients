import unittest

import numpy as np

import gradients

class GradientsTest(unittest.TestCase):
    def test_init_solid(self):
        result = gradients.init_solid(3, 2, (0, 0, 255))
        expected = np.array([
            0, 0, 255, 0, 0, 255, 0, 0, 255,
            0, 0, 255, 0, 0, 255, 0, 0, 255,
        ]).reshape((2, 3, 3))
        np.testing.assert_equal(result, expected)

    def test_linear_gradient(self):
        def linear(cur, total):
            return float(cur) / (total - 1)
        solid = gradients.init_solid(2, 3, (0, 0, 255))
        result = gradients.gradient(solid, (0, 255, 0), linear)
        expected = np.array([
            0, 0, 255, 0, 0, 255,
            0, 127.5, 127.5, 0, 127.5, 127.5,
            0, 255, 0, 0, 255, 0,
        ]).reshape((3, 2, 3))
        np.testing.assert_equal(result, expected)

    def test_quadratic_gradient(self):
        def quadratic(cur, total):
            return float(cur**2) / ((total - 1)**2)
        solid = gradients.init_solid(2, 3, (0, 0, 255))
        result = gradients.gradient(solid, (0, 255, 0), quadratic)
        expected = np.array([
            0, 0, 255, 0, 0, 255,
            0, 63.75, 191.25, 0, 63.75, 191.25,
            0, 255, 0, 0, 255, 0,
        ]).reshape((3, 2, 3))
        np.testing.assert_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
