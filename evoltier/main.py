# -*- coding: utf-8 -*-
from __future__ import print_function

import optimizer
import updater
import weight

from evoltier import model


def quad(x):
    #implementation of quadratic function.
    #global minima is zero.
    return (x * x).sum(axis=1)

def negative_quad(x):
    #implementation of quadratic function.
    #global mixima is zero.
    return - (x * x).sum(axis=1)


def main():
    # set probability distribution
    gaussian = model.MultiVariableGaussian(dim=3)
    
    # set utility function
    w = weight.QuantileBasedWeight(minimization=False)
    
    # set learning rate of distribution paramaters
    lr = {'mean': 1., 'var': 1. / (3 ** 2)}
    
    # set optimizer
    opt = optimizer.NaturalGradientOptimizer(gaussian, w, lr)
    
    # set updater
    upd = updater.Updater(optimizer=opt, obj_func=negative_quad, pop_size=7, threshold=-1e-6, out='result')
    
    # run IGO and print result
    print(upd.run())

if __name__ == '__main__':
    main()
