import numpy as np

from model import ProbabilityDistribution


class MultiVariableGaussian(ProbabilityDistribution):
    def __init__(self, dim, mean=None, cov=None, sigma=None, xp=np):
        self.xp = xp
        self.dim = dim
        self.sigma = sigma if sigma is not None else 0.3
        self.mean = mean if mean is not None else self.xp.random.random(self.dim)
        self.cov = cov if cov is not None else self.xp.identity(self.dim)
        self.model_class = 'FullCovariance'

    def sampling(self, pop_size):
        xp = self.xp
        sqrtC = xp.sqrt(self.eigan_vals)[:, None] * self.B.T
        samples = self.mean + self.sigma * xp.dot(xp.random.randn(pop_size, self.dim), sqrtC)
        return samples

    def get_info(self):
        return 'MaxEigenValue: {:.2e}, MinEigenValue: {:.2e},'.format(self.var_max, self.var_min)

    def get_info_dict(self):
        return {'MaxEigenValue': self.var_max, 'MinEigenValue': self.var_min}

    def generate_header(self):
        return ['MaxEigenValue', 'MinEigenValue']

    def eigan_decomp_cov(self, cov):
        xp = self.xp
        if xp == np:
            from numpy.dual import eigh
            eigan_vals, B = eigh(cov)
        else:
            eigan_vals, B = xp.linalg.eigh(cov)

        if eigan_vals.min() < 0:
            print('Wanning: Minimum eigan value is negative.')
            return False

        self.eigan_vals = eigan_vals
        self.B = B
        self.var_max = (self.sigma ** 2) * eigan_vals.max()
        self.var_min = (self.sigma ** 2) * eigan_vals.min()
        return True

    def use_gpu(self):
        super(MultiVariableGaussian, self).use_gpu()
        self.cov = self.xp.array(self.cov)
        self.mean = self.xp.array(self.mean)

    @property
    def cov(self):
        return self.__cov

    @cov.setter
    def cov(self, new_cov):
        if self.eigan_decomp_cov(new_cov):
            self.__cov = new_cov
