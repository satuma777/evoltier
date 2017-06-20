# -*- coding: utf-8 -*-

import numpy as np
from math import log, pi


class ProbabilityDistribution(object):
    def sampling(self):
        raise NotImplementedError()
    
    def get_param(self):
        raise NotImplementedError()

    def set_param(self):
        raise NotImplementedError()

    def log_likelihood(self):
        raise NotImplementedError()


class MultiVariableGaussian(ProbabilityDistribution):
    def __init__(self, dim, mean=None, covariance_matrix=None, stepsize=None, xp=np):
        self.dim = dim
        self.mean = mean
        self.covar_mat = covariance_matrix
        self.stepsize = stepsize
        self.xp = xp
        self.model_class = 'Gaussian'
        
        if self.mean is None:
            self.mean = xp.zeros(dim)
        if self.covar_mat is None:
            self.covar_mat = xp.identity(dim)
        if self.stepsize is None:
            self.stepsize = 1.
        
        assert self.mean.size == dim and self.covar_mat.size == dim * dim, \
            "Invalid value that dimensions DON'T match."
    
    def sampling(self, pop_size):
        sqrt_comat = self.xp.linalg.cholesky(self.covar_mat)
        sample = self.mean + np.dot(self.xp.random.normal(0., self.stepsize, (pop_size, self.dim)), sqrt_comat)
        return sample
    
    def get_param(self):
        return self.mean, self.covar_mat, self.stepsize
    
    def set_param(self, mean, covariance_matrix, stepsize):
        assert mean.size == dim and covariance_matrix.size == dim * dim, \
            "Invalid value that dimensions DON'T match."
        
        self.mean = mean
        self.covar_mat = covariance_matrix
        self.stepsize = stepsize
    
    def log_likelihood(self, sample):
        xp = self.xp
        pop_size = sample.shape[0]
        deviation = sample - self.mean
        comats = xp.array([self.stepsize * self.stepsize * self.covar_mat for _ in xrange(pop_size)])
        
        try:
            Cinv_der = xp.linalg.solve(comats, deviation)
            log_Cdet = log(xp.linalg.det(self.covar_mat))
        except AttributeError:
            Cinv_der = xp.array(np.linalg.solve(comats, deviation))
            log_Cdet = log(np.linalg.det(self.covar_mat))
        
        loglikelihood = -0.5 * (self.dim * log(2 * pi) + log_Cdet + xp.diag(xp.dot(Cinv_der, deviation.T)))
        
        return loglikelihood


if __name__ == '__main__':
    x = MultiVariableGaussian(1)
    s = x.sampling(3)
    lll = x.log_likelihood(s)
    print lll
