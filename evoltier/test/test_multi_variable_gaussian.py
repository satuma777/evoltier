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

    def test_get_param(self):
        mean, cov, sigma = self.gaussian.get_param()
        self.assertTrue(np.array_equal(mean, np.zeros(self.dim)))
        self.assertTrue(np.array_equal(cov, np.eye(self.dim)))
        self.assertEqual(sigma, 1.)

    def test_set_param(self):
        mean = [2 * np.zeros(self.dim), None]
        cov = [2 * np.eye(self.dim), None]
        sigma = [2, None]
        
        for m, v, s in itertools.product(mean, cov, sigma):
            set_gaussian = MultiVariableGaussian(self.dim)
            set_gaussian.set_param(m, v, s)
            if m is not None:
                self.assertTrue(np.array_equal(set_gaussian.mean, m))
            else:
                self.assertTrue(np.array_equal(set_gaussian.mean, self.gaussian.mean))
            
            if v is not None:
                self.assertTrue(np.array_equal(set_gaussian.cov, v))
            else:
                self.assertTrue(np.array_equal(set_gaussian.cov, self.gaussian.cov))
            
            if s is not None:
                self.assertEqual(set_gaussian.sigma, s)
            else:
                self.assertEqual(set_gaussian.sigma, self.gaussian.sigma)
            
    def test_calculate_log_likelihood(self):
        means = np.array([self.gaussian.mean])
        log_Cdet = np.log(np.linalg.det(self.gaussian.cov))
        min_lll = np.array([-0.5 * (self.dim * np.log(2 * np.pi) + log_Cdet)])
        self.assertTrue(np.array_equal(self.gaussian.calculate_log_likelihood(means), min_lll))


if __name__ == '__main__':
    unittest.main()
