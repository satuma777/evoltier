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

    def test_calculate_log_likelihood(self):
        with self.assertRaises(NotImplementedError):
            self.distribution.calculate_log_likelihood()

    def test_get_info(self):
        with self.assertRaises(NotImplementedError):
            self.distribution.get_info()

    def test_get_info_dict(self):
        with self.assertRaises(NotImplementedError):
            self.distribution.get_info_dict()

    def test_generate_header(self):
        with self.assertRaises(NotImplementedError):
            self.distribution.generate_header()

    def test_use_gpu(self):
        try:
            import cupy
        except ImportError:
            # this case is using by cpu only.
            with self.assertRaises(ImportError):
                self.distribution.use_gpu()
        else:
            # this case is using by gpu.
            self.assertTrue(isinstance(self.distribution.xp, cupy))


if __name__ == '__main__':
    unittest.main()
