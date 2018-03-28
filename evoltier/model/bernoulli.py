import numpy as np

from evoltier.model import ProbabilityDistribution


class Bernoulli(ProbabilityDistribution):
    def __init__(self, dim, theta=None, upper=None, lower=None, xp=np):
        self.dim = dim
        self.xp = xp
        self.upper = upper if upper is not None else 1. - 1. / self.dim
        self.lower = lower if lower is not None else 1. / self.dim
        self.theta = theta if theta is not None else 0.5 * self.xp.ones(self.dim)
        self.model_class = 'Bernoulli'
    
    def sampling(self, pop_size):
        xp = self.xp
        size = (pop_size, self.dim)
        samples = xp.random.binomial(n=1, p=self.theta, size=size)
        return samples
        
    def get_param(self):
        return self.theta
    
    def set_param(self, theta=None):
        if theta is not None:
            self.theta = theta
    
    def calculate_log_likelihood(self, sample):
        xp = self.xp
        lll = xp.sum(xp.log(sample * self.theta + (1. - sample) * (1. - self.theta)))
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

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, new_theta):
        xp = self.xp
        self. __theta = xp.minimum(xp.maximum(new_theta, self.lower), self.upper)
        print(self.__theta)
