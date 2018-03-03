from math import sqrt, floor, log
import numpy as np


class CMAESParameters(object):
    def __init__(self, dim, alpha_cov=2.):
        self.dim = dim
        self.alpha_cov = alpha_cov
    
    def get_parameters(self, weights, mu_eff=None, xp=np):
        dim = self.dim
        alpha_cov = self.alpha_cov

        if mu_eff is None:
            mu_eff = CMAESParameters.mu_eff(weights, xp=xp)

        # Hyperparameters for the covariance matrix adaptation
        c_C = self.c_C(mu_eff, dim)
        c_1 = self.c_1(mu_eff, dim, alpha_cov=alpha_cov)
        c_mu = self.c_mu(mu_eff, dim, c_1, alpha_cov=alpha_cov)

        # Hyperparameters for step-size control
        c_sigma = self.c_sigma(mu_eff, dim)
        d_sigma = self.d_sigma(mu_eff, dim, c_sigma)

        # Hyperparameter for the mean vector update
        c_m = self.c_m()

        return c_C, c_1, c_mu, c_sigma, d_sigma, c_m

    def get_population_size(self, scale=1):
        return scale * (4 + floor(3 * log(self.dim)))

    @staticmethod
    def mu_eff(weights, xp=np):
        return xp.linalg.norm(weights, ord=1) / xp.linalg.norm(weights)

    @staticmethod
    def c_C(mu_eff, dim):
        return (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)

    @staticmethod
    def c_1(mu_eff, dim, alpha_cov=2.):
        return alpha_cov / ((dim + 1.3) ** 2 + mu_eff)

    @staticmethod
    def c_mu(mu_eff, dim, c_1, alpha_cov=2., xp=np):
        return xp.min([1 - c_1, alpha_cov * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + alpha_cov * mu_eff / 2)])

    @staticmethod
    def c_sigma(mu_eff, dim):
        return (mu_eff + 2) / (dim + mu_eff + 5)

    @staticmethod
    def d_sigma(mu_eff, dim, c_sigma):
        return 1 + 2 * max([0, sqrt((mu_eff - 1) / (dim + 1)) - 1]) + c_sigma

    @staticmethod
    def c_m():
        return 1
