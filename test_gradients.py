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

    def test_init_gradient(self):
        result = gradients.init_gradient(
            2, 3, (0, 0, 255), (0, 255, 0), np.linspace
        )
        expected = np.array([
            0, 0, 255, 0, 0, 255,
            0, 127.5, 127.5, 0, 127.5, 127.5,
            0, 255, 0, 0, 255, 0,
        ]).reshape((3, 2, 3))
        np.testing.assert_equal(result, expected)

    def test_gradient(self):
        solid = gradients.init_solid(2, 3, (0, 0, 255))
        result = gradients.gradient(solid, (0, 255, 0))
        expected = np.array([
            0, 0, 255, 0, 0, 255,
            0, 127.5, 127.5, 0, 127.5, 127.5,
            0, 255, 0, 0, 255, 0,
        ]).reshape((3, 2, 3))
        np.testing.assert_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
