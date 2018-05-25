import unittest
import numpy as np
import itertools

from evoltier.model import MultiVariableGaussian


class TestMultiVariableGaussian(unittest.TestCase):

    def setUp(self):
        self.dim = 3
        self.gaussian = MultiVariableGaussian(self.dim)

    def tearDown(self):
        self.gaussian = None

    def test_sampling(self):
        pop_size = 100
        sample = self.gaussian.sampling(pop_size)
        self.assertTrue(sample.shape, (pop_size, self.dim))

if __name__ == '__main__':
    unittest.main()
