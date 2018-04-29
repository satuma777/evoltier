import numpy as np
from six.moves import range

from evoltier.model import ProbabilityDistribution


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

    def get_param(self):
        return self.mean, self.cov, self.sigma
    
    def set_param(self, mean=None, cov=None, sigma=None):
        if mean is not None:
            self.mean = mean
        
        if cov is not None:
            self.cov = cov
        
        if sigma is not None:
            self.sigma = sigma

    def calculate_log_likelihood(self, sample):
        xp = self.xp
        pop_size = len(sample)
        deviation = sample - self.mean
        comats = xp.array([self.sigma * self.cov for _ in range(pop_size)])
        Cinv_der = xp.linalg.solve(comats, deviation)
        log_Cdet = xp.log(xp.linalg.det(self.cov))
        loglikelihood = -0.5 * (self.dim * xp.log(2 * xp.pi) + log_Cdet + xp.diag(xp.dot(Cinv_der, deviation.T)))
        return loglikelihood

    def get_info(self):
        return 'MaxEigenValue: {}, MinEigenValue: {},'.format(self.var_max, self.var_min)

    def get_info_dict(self):
        return {'MaxEigenValue': self.var_max, 'MinEigenValue': self.var_min}

    def generate_header(self):
        return ['MaxEigenValue', 'MinEigenValue']
    
    def eigan_decomp_cov(self, cov):
        xp = self.xp
        if xp == np:
            from numpy.dual import eigh
            self.eigan_vals, self.B = eigh(cov)
        else:
            self.eigan_vals, self.B = xp.linalg.eigh(cov)
        self.var_min = self.sigma * self.sigma * self.eigan_vals.min()
        self.var_max = self.sigma * self.sigma * self.eigan_vals.max()

    def use_gpu(self):
        super(MultiVariableGaussian, self).use_gpu()
        self.cov = self.xp.array(self.cov)
        self.mean = self.xp.array(self.mean)

    @property
    def cov(self):
        return self.__cov
    
    @cov.setter
    def cov(self, new_cov):
        self.eigan_decomp_cov(new_cov)
        self.__cov = new_cov

