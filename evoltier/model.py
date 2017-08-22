# -*- coding: utf-8 -*-
import numpy as np
try:
    import cupy as cp
except ImportError:
    None
from math import log, pi
from six.moves import range


class ProbabilityDistribution(object):
    def __init__(self, xp=np):
        self.xp = xp
        raise NotImplementedError()
    
    def sampling(self):
        raise NotImplementedError()
    
    def get_param(self):
        raise NotImplementedError()

    def set_param(self):
        raise NotImplementedError()

    def log_likelihood(self):
        raise NotImplementedError()
    
    def get_info(self):
        raise NotImplementedError()
    
    def get_info_dict(self):
        raise NotImplementedError()
    
    def generate_header(self):
        raise NotImplementedError()

    def use_gpu(self):
        self.xp = cp


class MultiVariableGaussian(ProbabilityDistribution):
    def __init__(self, dim, mean=None, var=None, stepsize=None, xp=np):
        self.dim = dim
        self.mean = mean
        self.var = var
        self.stepsize = stepsize
        self.xp = xp
        self.model_class = 'Gaussian'
        
        if self.mean is None:
            self.mean = xp.zeros(dim)
        if self.var is None:
            self.var = xp.identity(dim)
        if self.stepsize is None:
            self.stepsize = 1.
        
        assert self.mean.size == dim and self.var.size == dim * dim, \
            "Invalid value that dimensions DON'T match."
    
    def sampling(self, pop_size):
        xp = self.xp
        m = self.mean
        var = self.stepsize * self.var
        sample = xp.random.multivariate_normal(m, var, pop_size)
        return sample
    
    def get_param(self):
        return self.mean, self.var, self.stepsize
    
    def set_param(self, mean=None, var=None, stepsize=None):
        dim = self.dim
        xp = self.xp
        
        assert mean.size == dim and var.size == dim ** 2, \
            "Invalid value that dimensions DON'T match."
        
        if mean is None:
            self.mean = xp.zeros(dim)
        else:
            self.mean = mean
            
        if var is None:
            self.var = xp.identity(dim)
        else:
            self.var = var
            
        if stepsize is None:
            self.stepsize = 1.
        else:
            self.stepsize = stepsize
    
    def calculate_log_likelihood(self, sample):
        xp = self.xp
        pop_size = sample.shape[0]
        deviation = sample - self.mean
        comats = xp.array([self.stepsize * self.stepsize * self.var for _ in range(pop_size)])
        
        try:
            Cinv_der = xp.linalg.solve(comats, deviation)
            log_Cdet = log(xp.linalg.det(self.var))
        except AttributeError:
            Cinv_der = xp.array(np.linalg.solve(comats, deviation))
            log_Cdet = log(np.linalg.det(self.var))
        
        loglikelihood = -0.5 * (self.dim * log(2 * pi) + log_Cdet + xp.diag(xp.dot(Cinv_der, deviation.T)))
        
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