from __future__ import print_function, division

from evoltier.optimizer import Optimizer
from evoltier.model.multi_variable_gassian import MultiVariableGaussian


class GaussianNaturalGradientOptimizer(Optimizer):
    def __init__(self, weight_function, lr, distribution=None, dim=None):
        if dim is None and distribution is None:
            print('Need to set argument "dim" or "distribution"')
            raise
        dist = distribution if distribution is not None else MultiVariableGaussian(dim=dim)
        super(GaussianNaturalGradientOptimizer, self).__init__(dist, weight_function, lr)

    def update(self, evals, sample):
        self.t += 1
        xp = self.target.xp
        weights = self.w_func(evals, xp=xp)
        self.lr.set_parameters(weights, xp=xp)

        mean, cov, sigma = self.target.mean, self.target.cov, self.target.sigma
        grad_m, grad_cov = self.compute_natural_grad(weights, sample, mean, cov, sigma)

        self.target.mean += self.lr.c_m * grad_m
        self.target.cov += self.lr.c_C * grad_cov

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
        info = {'LearningRateMean': self.lr.c_m,
                'LearningRateCov': self.lr.c_C}
        return info
