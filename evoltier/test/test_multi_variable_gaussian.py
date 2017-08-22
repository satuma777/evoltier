import unittest
import copy
import numpy as np
from scipy.stats import sem, chi2_contingency
import itertools

from evoltier.model import MultiVariableGaussian


class TestMultiVariableGaussian(unittest.TestCase):

    def setUp(self):
        self.dim = 3
        self.gaussian = MultiVariableGaussian(self.dim)

    def tearDown(self):
        self.gaussian = None

    def test_sampling(self):
        pop_size = 1000
        sample = self.gaussian.sampling(pop_size)
        
        self.assertEqual(np.size(sample), pop_size * self.dim)
        
        # 99% confidence interval estimation using standard error
        estimate_mean = np.mean(sample, axis=0)
        standard_error = 2.58 * sem(sample, axis=0)
        check_lower = self.gaussian.mean > (estimate_mean - standard_error)
        check_upper = self.gaussian.mean < (estimate_mean + standard_error)
        self.assertTrue(np.all([check_lower, check_upper]))
        
        # HACK: confidence interval estimation for covariance matrix
        estimate_var = np.cov(sample.T)
        self.assertTrue(np.all(np.abs(estimate_var - self.gaussian.var) ** 2 < 0.01))

    def test_get_param(self):
        mean, var, stepsize = self.gaussian.get_param()
        self.assertTrue(np.array_equal(mean, np.zeros(self.dim)))
        self.assertTrue(np.array_equal(var, np.eye(self.dim)))
        self.assertEqual(stepsize, 1.)

    def test_set_param(self):
        mean = [2 * np.zeros(self.dim), None]
        var = [2 * np.eye(self.dim), None]
        stepsize = [2, None]
        
        for m, v, s in itertools.product(mean, var, stepsize):
            set_gaussian = MultiVariableGaussian(self.dim)
            set_gaussian.set_param(m, v, s)
            if m is not None:
                self.assertTrue(np.array_equal(set_gaussian.mean, m))
            else:
                self.assertTrue(np.array_equal(set_gaussian.mean, self.gaussian.mean))
            
            if v is not None:
                self.assertTrue(np.array_equal(set_gaussian.var, v))
            else:
                self.assertTrue(np.array_equal(set_gaussian.var, self.gaussian.var))
            
            if s is not None:
                self.assertEqual(set_gaussian.stepsize, s)
            else:
                self.assertEqual(set_gaussian.stepsize, self.gaussian.stepsize)
            
    def test_calculate_log_likelihood(self):
        means = np.array([self.gaussian.mean])
        log_Cdet = np.log(np.linalg.det(self.gaussian.var))
        min_lll = np.array([-0.5 * (self.dim * np.log(2 * np.pi) + log_Cdet)])
        self.assertTrue(np.array_equal(self.gaussian.calculate_log_likelihood(means), min_lll))


if __name__ == '__main__':
    unittest.main()
