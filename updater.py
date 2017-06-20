import numpy as np
import collections
from math import log, pi


class Updater(object):
    """Base class of all updater."""
    def setup(self, distribution):
        self.target = distribution
        self.t = 0
        self._hooks = collections.OrderedDict()

    def update(self):
        raise NotImplementedError()

    def add_hook(self, hook, name=None):
        if not callable(hook):
            raise TypeError('hook function is not callable')
        if not hasattr(self, '_hooks'):
            raise RuntimeError('call `setup` method before `add_hook` method')

        if name is None:
            name = hook.name
        if name in self._hooks:
            raise KeyError('hook %s already exists' % name)
        
        self._hooks[name] = hook

    def call_hooks(self):
        """Invokes hook functions in registration order."""
        for hook in self._hooks.itervalues():
            self._call_hook(hook)

    def _call_hook(self, hook):
        hook(self, self.target.get_param())
  

class NaturalGradientUpdater(Updater):
    def __init__(self):
        super(Updater, self).__init__()
    
    def update(self, weight, sample):
        self.call_hooks()
        self.t += 1
        
        if self.target.model_class in 'Gaussian':
            self.gaussian_param_update(weight, sample)
        if self.target.model_class in 'Bernoulli':
            self.bernouil_param_update(weight, sample)
    
    def gaussian_param_update(self, weight, sample):
        grad_m, grad_C = self._compute_natural_grad_gaussian(weight, sample)
        
        pass
    
    def bernouil_param_update(self, weight, sample):
        pass

    def _compute_natural_grad_gaussian(self, weight, sample):
        mean = self.target.mean
        covar_mat = self.target.covar_mat
        
        derivation = sample - mean
        w_der = weight * derivation.T
        grad_m = w_der.sum(axis=1)

        if self.target.model_class in 'Isotropic':
            grad_C =
        elif self.target.model_class in 'Separable':
            grad_C =
        else:
            grad_C = np.dot(w_der, derivation) - weight.sum() * covar_mat
        
        return grad_m, grad_C