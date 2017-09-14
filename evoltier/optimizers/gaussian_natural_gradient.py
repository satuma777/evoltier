from __future__ import print_function, division

from evoltier.optimizer import Optimizer


class GaussianNaturalGradientOptimizer(Optimizer):
    def __init__(self, distribution, weight_function, lr):
        super(GaussianNaturalGradientOptimizer, self).__init__(distribution, weight_function, lr)
        
        if 'mean' not in lr or 'var' not in lr:
            print('lr does not have attribute "mean" or "var". ')
            exit(1)
    
    def update(self, evals, sample):
        # TODO:Implementaion of Comulative Step-size Update
        # TODO:Implementaion of rank-one Update
        self.t += 1
        weight = self.w_func(evals)

        mean, var, stepsize = self.target.get_param()
        grad_m, grad_var = self.compute_natural_grad(weight, sample, mean, var)

        new_mean = mean + self.lr['mean'] * grad_m
        new_var = var + self.lr['var'] * grad_var

        self.target.set_param(mean=new_mean, var=new_var)
    
    def compute_natural_grad(self, weight, sample, mean, var):
        xp = self.target.xp
        derivation = sample - mean
        w_der = weight * derivation.T
        grad_m = w_der.sum(axis=1)
        
        if self.target.model_class in 'Isotropic':
            norm_w_der = xp.diag(xp.dot(w_der, w_der.T))
            grad_var = (xp.sum(weight * norm_w_der) / self.target.dim) - (xp.sum(weight) * var)
        elif self.target.model_class in 'Separable':
            grad_var = (w_der * derivation.T).sum(axis=1) - (weight.sum() * var)
        else:
            grad_var = xp.dot(w_der, derivation) - weight.sum() * var
        
        return grad_m, grad_var
    
    def generate_header(self):
        header = ['LearningRateMean', 'LearningRateVar']
        return header
    
    def get_info_dict(self):
        info = {'LearningRateMean': self.lr['mean'], 'LearningRateVar': self.lr['var']}
        return info
