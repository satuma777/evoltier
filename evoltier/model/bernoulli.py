import numpy as np

from evoltier.model import ProbabilityDistribution


class Bernoulli(ProbabilityDistribution):
    def __init__(self, dim, theta=None, xp=np):
        self.dim = dim
        self.theta = theta
        self.xp = xp
        self.model_class = 'Bernoulli'
    
        if self.theta is None:
            self.theta = self.xp.ones(self.dim)
            
        assert self.theta.size == self.dim, \
            "Invalid value that dimensions DON'T match."
    
    def sampling(self, pop_size):
        xp = self.xp
        size = (pop_size,) + self.theta.shape
        samples = xp.random.binomial(n=1, p=self.theta, size=size)
        return samples
        
    def get_param(self):
        return self.theta
    
    def set_param(self, theta=None):
        if theta is not None:
            self.theta = theta
        
        assert self.theta.size == self.dim, \
            "Invalid value that dimensions DON'T match."
    
    def calculate_log_likelihood(self, sample):
        xp = self.xp
        lll = xp.sum(xp.log(sample * self.theta + (1. - sample) * (1. - self.sample)))
        return lll
    
    def get_info(self):
        mean, var, median, mini, maxi = self._calculate_stat()
        string_info = 'Mean: {}, Variance: {}, Median: {}, Min: {}, Max: {}'.format(mean, var, median, mini, maxi)
        return string_info
    
    def get_info_dict(self):
        mean, var, median, mini, maxi = self._calculate_stat()
        dict_info = {'Mean': mean, 'Variance': var, 'Median': median, 'Min': mini, 'Max': maxi}
        return dict_info
    
    def generate_header(self):
        return ['Mean', 'Variance', 'Median', 'Min', 'Max']
    
    def _calculate_stat(self):
        xp = self.xp
        mean = xp.mean(self.theta)
        var = xp.var(self.theta)
        median = xp.median(self.theta)
        mini = xp.min(self.theta)
        maxi = xp.max(self.theta)
        return mean, var, median, mini, maxi