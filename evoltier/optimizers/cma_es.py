from __future__ import print_function, division
from math import sqrt

from evoltier.optimizers import GaussianNaturalGradientOptimizer


class CMAES(GaussianNaturalGradientOptimizer):
    def __init__(self, distribution, weight_function, lr):
        super(CMAES, self).__init__(distribution, weight_function, lr)

        if ('c_1' not in lr) or ('c_sigma' not in lr)
        or ('d_sigma' not in lr) or ('c_C' not in lr) or :
            print('lr does not have attribute "c_1", "c_sigma", "d_sigma" or "c_C".')
            exit(1)

        xp = self.target.xp
        dim = self.target.dim
        self.p_c = xp.zeros(dim)
        self.p_sigma = xp.zeros(dim)
        self.ex_norm = sqrt(dim) * (1. - 1. / (4. * dim) + 1. / (21. * dim ** 2))

    def update(self, evals, sample):
        self.t += 1
        weight = self.w_func(evals)

        mean, cov, sigma = self.target.get_param()
        grad_m, grad_cov = self.compute_natural_grad(weight, sample, mean, cov, sigma)
        self.p_c, self.p_sigma, h_sigma = self.compute_evolutionary_path(weight, grad_m)
        rank_one_cov = self.lr['c_1'] * self.p_c * self.target.xp.outer(self.p_c, self.p_c)
        delta = (1. - h_sigma) * self.lr['c_C'] * (2. - self.lr['c_C'])
        cov_factor = (1 + self.lr['c_1'] * (delta - 1.) - self.lr['cov'])

        new_mean = mean + self.lr['mean'] * grad_m
        new_cov = cov_factor * cov + self.lr['cov'] * grad_cov + rank_one_cov
        new_sigma = sigma * self.compute_step_size(self.p_sigma)

        self.target.set_param(mean=new_mean, cov=new_cov, sigma=new_sigma)

    def compute_evolutionary_path(self, weight, grad_m):
        xp = self.target.xp
        p_sigma_norm_scaled = xp.linalg.norm(self.p_sigma) /
            sqrt(1. - (1. - self.lr['c_sigma']) ** (2 * (self.t + 1)))
        h_sigma = self._heaviside(self.ex_norm, p_sigma_norm_scaled)
        mu_eff = 1. / xp.dot(weight, weight)
        p_c = (1. - self.lr['c_C']) * self.p_c +
            h_sigma * sqrt(self.lr['c_C'] * (2. - self.lr['c_C']) * mu_eff) * grad_m
        inv_sqrtC = xp.linalg.inv(xp.linalg.cholesky(self.target.cov))
        p_sigma = (1. - self.lr['c_sigma']) * self.p_sigma +
            h_sigma * sqrt(self.lr['c_sigma'] * (2. - self.lr['c_sigma']) * mu_eff) *
            inv_sqrtC * grad_m
        return p_c, p_sigma, h_sigma

    def compute_step_size(self, p_sigma):
        xp = self.target.xp
        control_factor = xp.exp(self.lr['c_sigma'] / self.lr['d_sigma']
                            * (-1. + xp.linalg.norm(self.p_sigma) / self.ex_norm))
        return control_factor

    def _heaviside(x, y):
        if x < y : return 0.
        else: return 1.
