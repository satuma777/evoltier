from __future__ import print_function, division

from evoltier.optimizer import Optimizer
from evoltier.utils import CMAESParameters


class GaussianNaturalGradientOptimizer(Optimizer):
    def __init__(self, distribution, weight_function, lr):
        super(GaussianNaturalGradientOptimizer, self).__init__(distribution, weight_function, lr)

    def update(self, evals, sample):
        self.t += 1
        weights = self.w_func(evals, xp=self.target.xp)

        mean, cov, sigma = self.target.mean, self.target.cov, self.target.sigma
        lr_m, lr_cov = self.get_hyperparams(self.lr, weights)
        grad_m, grad_cov = self.compute_natural_grad(weights, sample, mean, cov, sigma)

        self.target.mean += lr_m * grad_m
        self.target.cov += lr_cov * grad_cov

    def compute_natural_grad(self, weight, sample, mean, cov, sigma):
        xp = self.target.xp
        derivation = (sample - mean) / sigma
        w_der = weight * derivation.T
        grad_m = sigma * w_der.sum(axis=1)

        if self.target.model_class == 'Isotropic':
            norm_w_der = xp.diag(xp.dot(w_der, w_der.T))
            grad_cov = (xp.sum(weight * norm_w_der) / self.target.dim) - (xp.sum(weight) * cov)
        elif self.target.model_class == 'Separable':
            grad_cov = (w_der * derivation.T).sum(axis=1) - (weight.sum() * cov)
        else:
            grad_cov = xp.dot(w_der, derivation) - weight.sum() * cov

        return grad_m, grad_cov

    def generate_header(self):
        header = ['LearningRateMean', 'LearningRateCov']
        return header

    def get_info_dict(self):
        info = {'LearningRateMean': self.lr['mean'],
                'LearningRateCov': self.lr['cov']}
        return info

    def get_hyperparams(self, lr, weights):
        if isinstance(lr, CMAESParameters):
            mu_eff = lr.mu_eff(weights, xp=self.target.xp)
            return lr.c_m(), lr.c_mu(mu_eff, self.target.dim, 0)
        else:
            return self.lr['mean'], self.lr['cov']