# -*- coding: utf-8 -*-
from __future__ import print_function, division


class Optimizer(object):
    """Base class of all optimizer."""
    
    def __init__(self, distribution, weight_function, lr):
        self.target = distribution
        self.t = 0
        self.w_func = weight_function
        self.lr = lr
        
        if not callable(self.w_func):
            raise TypeError('weight function is NOT callable.')
    
    def update(self):
        raise NotImplementedError()
    
    def get_info_dict(self):
        raise NotImplementedError()
    
    def generate_header(self):
        raise NotImplementedError()


class NaturalGradientOptimizer(Optimizer):
    def __init__(self, distribution, weight_function, lr):
        self.target = distribution
        self.t = 0
        self.w_func = weight_function
        self.lr = lr
        
        if 'mean' not in lr or 'var' not in lr:
            print('lr does not have attribute "mean" or "var". ')
            exit(1)
    
    def update(self, evals, sample):
        self.t += 1
        
        weight = self.w_func(evals)
        
        if self.target.model_class in 'Gaussian':
            self.gaussian_param_update(weight, sample)
        if self.target.model_class in 'Bernoulli':
            self.bernouil_param_update(weight, sample)
    
    def gaussian_param_update(self, weight, sample):
        # TODO:Implementaion of Comulative Step-size Update
        # TODO:Implementaion of rank-one Update
        
        mean, var, stepsize = self.target.get_param()
        grad_m, grad_var = self.compute_natural_grad_gaussian(weight, sample, mean, var)
        
        new_mean = mean + self.lr['mean'] * grad_m
        new_var = var + self.lr['var'] * grad_var
        
        self.target.set_param(new_mean, new_var)
    
    def bernouil_param_update(self, weight, sample):
        # TODO:Implementaion of Population-based increment learning
        pass
    
    def compute_natural_grad_gaussian(self, weight, sample, mean, var):
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
            grad_var = xp.dot(w_der, derivation) - (weight.sum() * var)
        
        return grad_m, grad_var

    def generate_header(self):
        if self.target.model_class in 'Gaussian':
            header = ['LearningRateMean', 'LearningRateVar']
        else:
            header = ['LearningRate']
        
        return header
    
    def get_info_dict(self):
        if self.target.model_class in 'Gaussian':
            info = {'LearningRateMean': self.lr['mean'], 'LearningRateVar': self.lr['var']}
        else:
            info = {'LearningRate': self.lr}
            
        return info
