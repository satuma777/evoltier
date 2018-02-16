from __future__ import print_function, division

from evoltier.optimizer import Optimizer


class GaussianNaturalGradientOptimizer(Optimizer):
    def __init__(self, distribution, weight_function, lr):
        super(GaussianNaturalGradientOptimizer, self).__init__(distribution, weight_function, lr)

        if 'mean' not in lr or 'cov' not in lr:
            print('lr does not have attribute "mean" or "cov". ')
            exit(1)

    def update(self, evals, sample):
        self.t += 1
        weight = self.w_func(evals)

        mean, cov, sigma = self.target.get_param()
        grad_m, grad_cov = self.compute_natural_grad(weight, sample, mean, cov, sigma)

        new_mean = mean + self.lr['mean'] * grad_m
        new_cov = cov + self.lr['cov'] * grad_cov

        self.target.set_param(mean=new_mean, cov=new_cov)

    def compute_natural_grad(self, weight, sample, mean, cov, sigma):
        xp = self.target.xp
        derivation = (sample - mean) / sigma
        w_der = weight * derivation.T
        grad_m = w_der.sum(axis=1)

        if self.target.model_class in 'Isotropic':
            norm_w_der = xp.diag(xp.dot(w_der, w_der.T))
            grad_cov = (xp.sum(weight * norm_w_der) / self.target.dim) - (xp.sum(weight) * cov)
        elif self.target.model_class in 'Separable':
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
