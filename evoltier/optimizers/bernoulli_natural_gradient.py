from __future__ import print_function, division

from evoltier.optimizer import Optimizer


class BernoulliNaturalGradientOptimizer(Optimizer):
    def __init__(self, distribution, weight_function, lr):
        super(BernoulliNaturalGradientOptimizer, self).__init__(distribution, weight_function, lr)

    def update(self, evals, sample):
        self.t += 1
        weights = self.w_func(evals, xp=self.target.xp)
        self.lr.set_parameters(weights, xp=self.target.xp)

        grad_theta = self.compute_natural_grad(weights, sample, self.target.theta)
        self.target.theta += self.lr.eta * grad_theta

    def compute_natural_grad(self, weights, sample, theta):
        xp = self.target.xp
        grad_theta = xp.sum(weights[:, None] * (sample - theta), axis=0)
        return grad_theta

    def get_info_dict(self):
        info = {'LearningRate': self.lr.eta}
        return info

    def generate_header(self):
        header = ['LearningRate']
        return header

