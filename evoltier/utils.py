from math import sqrt, floor, log
import numpy as np


class HyperParameters(object):
    def __init__(self, dict_params):
        self.dict_params = dict_params

    def set_parameters(self, *args, **kargs):
        for name, value in self.dict_params.items():
            setattr(self, name, value)
        return True


class CMAESParameters(object):
    def __init__(self, dim, alpha_cov=2.):
        self.dim = dim
        self.alpha_cov = alpha_cov
    
    def set_parameters(self, weights, xp=np):
        dim = self.dim
        alpha_cov = self.alpha_cov

        self.mu_eff = self.compute_mu_eff(weights, xp=xp)

        # Hyperparameters for the covariance matrix adaptation
        self.c_C = self.compute_c_C(self.mu_eff, dim)
        self.c_1 = self.compute_c_1(self.mu_eff, dim, alpha_cov=alpha_cov)
        self.c_mu = self.compute_c_mu(self.mu_eff, dim, self.c_1, alpha_cov=alpha_cov, xp=xp)

        # Hyperparameters for step-size control
        self.c_sigma = self.compute_c_sigma(self.mu_eff, dim)
        self.d_sigma = self.compute_d_sigma(self.mu_eff, dim, self.c_sigma)

        # Hyperparameter for the mean vector update
        self.c_m = self.compute_c_m()

        return True

    def get_population_size(self, scale=1):
        return scale * (4 + floor(3 * log(self.dim)))

    @staticmethod
    def compute_mu_eff(weights, xp=np):
        return xp.linalg.norm(weights, ord=1) / xp.linalg.norm(weights)

    @staticmethod
    def compute_c_C(mu_eff, dim):
        return (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)

    @staticmethod
    def compute_c_1(mu_eff, dim, alpha_cov=2.):
        return alpha_cov / ((dim + 1.3) ** 2 + mu_eff)

    @staticmethod
    def compute_c_mu(mu_eff, dim, c_1, alpha_cov=2., xp=np):
        return xp.min([1 - c_1, alpha_cov * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + alpha_cov * mu_eff / 2)])

    @staticmethod
    def compute_c_sigma(mu_eff, dim):
        return (mu_eff + 2) / (dim + mu_eff + 5)

    @staticmethod
    def compute_d_sigma(mu_eff, dim, c_sigma):
        return 1 + 2 * max([0, sqrt((mu_eff - 1) / (dim + 1)) - 1]) + c_sigma

    @staticmethod
    def compute_c_m():
        return 1
