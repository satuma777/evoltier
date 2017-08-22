import unittest

from evoltier.model import MultiVariableGaussian


class TestMultiVariableGaussian(unittest.TestCase):

    def setUp(self):
        self.gaussian = MultiVariableGaussian(1)

    def tearDown(self):
        self.gaussian = None

    def test_sampling(self):
        self.assertEqual(len(self.gaussian.sampling(10)), 10)

    def test_get_param(self):
        mean, var, stepsize = self.gaussian.get_param()
        self.assertEqual(mean, [0.])
        self.assertEqual(var, [[1.]])
        self.assertEqual(stepsize, 1.0)

    def test_set_param(self):
        with self.assertRaises(AttributeError):
            self.gaussian.set_param()

    def test_calculate_log_likelihood(self):
        sample = self.gaussian.sampling(3)
        sample -= sample
        self.assertEqual(len(self.gaussian.calculate_log_likelihood(sample)), 3)

    def test_use_gpu(self):
        with self.assertRaises(NameError):
            self.gaussian.use_gpu()

if __name__ == '__main__':
    unittest.main()
