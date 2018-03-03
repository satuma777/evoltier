from __future__ import print_function, division

from evoltier.optimizers.gaussian_natural_gradient import GaussianNaturalGradientOptimizer
from evoltier.utils import CMAESParameters


class CMAES(GaussianNaturalGradientOptimizer):
    def __init__(self, distribution, weight_function, lr):
        super(CMAES, self).__init__(distribution, weight_function, lr)

        xp = self.target.xp
        dim = self.target.dim
        self.p_c = xp.zeros(dim)
        self.p_sigma = xp.zeros(dim)
        self.ex_norm = xp.sqrt(dim) * (1. - 1. / (4. * dim) + 1. / (21. * (dim ** 2)))
        self.__cache_hyperparams = False

    def update(self, evals, sample):
        xp = self.target.xp
        self.t += 1
        weight = self.w_func(evals, xp=xp)

        mean, cov, sigma = self.target.mean, self.target.cov, self.target.sigma

        mu_eff = CMAESParameters.mu_eff(weight, xp=xp)
        c_C, c_1, c_mu, c_sigma, d_sigma, c_m = self.get_hyperparams(self.lr, mu_eff)
        grad_m, grad_cov = self.compute_natural_grad(weight, sample, mean, cov, sigma)

        h_sigma = self.update_evo_path(mu_eff, grad_m / sigma, c_sigma, c_C)
        delta = (1. - h_sigma) * c_C * (2. - c_C)

        self.target.mean += c_m * grad_m
        self.target.cov += c_1 * (cov * (delta - 1) + xp.outer(self.p_c, self.p_c)) \
                           + c_mu * (grad_cov - xp.sum(weight) * cov)
        self.target.sigma *= self.compute_step_size(c_sigma, d_sigma)

    def update_evo_path(self, mu_eff, y_w, c_sigma, c_C):
        xp = self.target.xp
        
        # compute new p_sigma
        D_inv = xp.reciprocal(xp.sqrt(self.target.eigan_vals))[:, None]
        inv_sqrtC = xp.dot(self.target.B * D_inv, self.target.B.T)
        self.p_sigma += xp.sqrt(c_sigma * (2. - c_sigma) * mu_eff) * xp.dot(y_w, inv_sqrtC) - c_sigma * self.p_sigma

        # compute new p_c
        p_sigma_norm_scaled = xp.linalg.norm(self.p_sigma) / xp.sqrt(1. - ((1. - c_sigma) ** (2 * (self.t + 1))))
        h_sigma = (1.4 + 2. / (self.target.dim + 1)) * self.ex_norm < p_sigma_norm_scaled
        self.p_c += h_sigma * xp.sqrt(c_C * (2. - c_C) * mu_eff) * y_w - c_C * self.p_c
        return h_sigma

    def compute_step_size(self, c_sigma, d_sigma):
        xp = self.target.xp
        control_factor = xp.exp((c_sigma / d_sigma) * (-1. + xp.linalg.norm(self.p_sigma) / self.ex_norm))
        return control_factor
    
    def get_hyperparams(self, lr, *args):
        if isinstance(lr, CMAESParameters):
            return lr.get_parameters(None, mu_eff=args[0], xp=self.target.xp)
        else:
            c_m = self.lr['mean']
            c_mu = self.lr['cov']
            c_1 = self.lr['c_1']
            c_C = self.lr['c_C']
            c_sigma = self.lr['c_sigma']
            d_sigma = self.lr['d_sigma']
        return c_C, c_1, c_mu, c_sigma, d_sigma, c_m

    @staticmethod
    def _heaviside(x, y):
        if x < y:
            return 0.
        else:
            return 1.
