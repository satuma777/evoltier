import numpy as np
from six.moves import range

from evoltier.model import ProbabilityDistribution


class MultiVariableGaussian(ProbabilityDistribution):
    def __init__(self, dim, mean=None, var=None, stepsize=None, xp=np):
        self.dim = dim
        self.mean = mean
        self.var = var
        self.stepsize = stepsize
        self.xp = xp
        self.model_class = 'Gaussian'
        
        if self.mean is None:
            self.mean = self.xp.zeros(self.dim)
            
        if self.var is None:
            self.var = self.xp.identity(self.dim)
            
        if self.stepsize is None:
            self.stepsize = 1.
        
        assert self.mean.size == self.dim and self.var.size == self.dim ** 2, \
            "Invalid value that dimensions DON'T match."
    
    def sampling(self, pop_size):
        xp = self.xp
        m = self.mean
        var = self.stepsize * self.var
        sample = xp.random.multivariate_normal(mean=m, cov=var, size=pop_size, check_valid='raise')
        return sample
    
    def get_param(self):
        return self.mean, self.var, self.stepsize
    
    def set_param(self, mean=None, var=None, stepsize=None):
        if mean is not None:
            self.mean = mean
        
        if var is not None:
            self.var = var
        
        if stepsize is not None:
            self.stepsize = stepsize

        assert self.mean.size == self.dim and self.var.size == self.dim ** 2, \
            "Invalid value that dimensions DON'T match."
    
    def calculate_log_likelihood(self, sample):
        xp = self.xp
        pop_size = len(sample)
        deviation = sample - self.mean
        comats = xp.array([self.stepsize * self.var for _ in range(pop_size)])
        
        try:
            Cinv_der = xp.linalg.solve(comats, deviation)
            log_Cdet = xp.log(xp.linalg.det(self.var))
        except AttributeError:
            Cinv_der = xp.array(np.linalg.solve(comats, deviation))
            log_Cdet = xp.log(np.linalg.det(self.var))
        
        loglikelihood = -0.5 * (self.dim * xp.log(2 * xp.pi) + log_Cdet + xp.diag(xp.dot(Cinv_der, deviation.T)))
        
        return loglikelihood
    
    def get_info(self):
        xp = self.xp
        eigan_val = xp.linalg.eigvalsh(self.var)
        string_eigen = 'MaxEigenValue: {}, MinEigenValue: {},'.format(eigan_val.max(), eigan_val.min())
        return string_eigen
    
    def get_info_dict(self):
        xp = self.xp
        eigan_val = xp.linalg.eigvalsh(self.var)
        info = {'MaxEigenValue': eigan_val.max(), 'MinEigenValue': eigan_val.min()}
        return info
    
    def generate_header(self):
        return ['MaxEigenValue', 'MinEigenValue']
