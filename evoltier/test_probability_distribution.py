import unittest

from evoltier.model import ProbabilityDistribution


class TestProbabilityDistribution(unittest.TestCase):

    def setUp(self):
        self.distribution = ProbabilityDistribution()

    def tearDown(self):
        self.distribution = None

    def test_sampling(self):
        with self.assertRaises(NotImplementedError):
            self.distribution.sampling()

    def test_get_param(self):
        with self.assertRaises(NotImplementedError):
            self.distribution.get_param()

    def test_set_param(self):
        with self.assertRaises(NotImplementedError):
            self.distribution.set_param()

    def test_log_likelihood(self):
        with self.assertRaises(NotImplementedError):
            self.distribution.log_likelihood()


if __name__ == '__main__':
    unittest.main()
