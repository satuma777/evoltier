import numpy as np
import multiprocessing as mp
from six.moves import range

from evoltier.model import ProbabilityDistribution


def _sampling_gauss(arr):
    np.random.seed()
    return np.random.normal(size=arr)
    

class MultiVariableGaussian(ProbabilityDistribution):
    def __init__(self, dim, mean=None, cov=None, sigma=None, xp=np):
        self.dim = dim
        self.mean = mean
        self.cov = cov
        self.sigma = sigma
        self.xp = xp
        self.model_class = 'Gaussian'
        
        if self.mean is None:
            self.mean = self.xp.zeros(self.dim)
            
        if self.cov is None:
            self.cov = self.xp.identity(self.dim)
            
        if self.sigma is None:
            self.sigma = 1.
        
        assert self.mean.size == self.dim and self.cov.size == self.dim ** 2, \
            "Invalid value that dimensions DON'T match."
    
    def sampling(self, pop_size):
        xp = self.xp
        m = self.mean
        cov = self.sigma * self.cov
        samples = xp.random.multivariate_normal(mean=m, cov=cov, size=pop_size, check_valid='raise')
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

        assert self.mean.size == self.dim and self.cov.size == self.dim ** 2, \
            "Invalid value that dimensions DON'T match."
    
    def calculate_log_likelihood(self, sample):
        xp = self.xp
        pop_size = len(sample)
        deviation = sample - self.mean
        comats = xp.array([self.sigma * self.cov for _ in range(pop_size)])
        
        try:
            Cinv_der = xp.linalg.solve(comats, deviation)
            log_Cdet = xp.log(xp.linalg.det(self.cov))
        except AttributeError:
            Cinv_der = xp.array(np.linalg.solve(comats, deviation))
            log_Cdet = xp.log(np.linalg.det(self.cov))
        
        loglikelihood = -0.5 * (self.dim * xp.log(2 * xp.pi) + log_Cdet + xp.diag(xp.dot(Cinv_der, deviation.T)))
        
        return loglikelihood
    
    def get_info(self):
        xp = self.xp
        eigan_val = xp.linalg.eigvalsh(self.cov)
        string_eigen = 'MaxEigenValue: {}, MinEigenValue: {},'.format(eigan_val.max(), eigan_val.min())
        return string_eigen
    
    def get_info_dict(self):
        xp = self.xp
        eigan_val = xp.linalg.eigvalsh(self.cov)
        info = {'MaxEigenValue': eigan_val.max(), 'MinEigenValue': eigan_val.min()}
        return info
    
    def generate_header(self):
        return ['MaxEigenValue', 'MinEigenValue']
