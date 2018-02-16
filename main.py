# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np


from evoltier.optimizers import GaussianNaturalGradientOptimizer
from evoltier import updater
from evoltier import weight
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
    gaussian = model.MultiVariableGaussian(dim=100)

    # set utility function
    w = weight.QuantileBasedWeight(minimization=True)

    # set learning rate of distribution paramaters
    lr = {'mean': 1., 'cov': 1. / (3 ** 2)}

    # set optimizer
    opt = GaussianNaturalGradientOptimizer(gaussian, w, lr)

    # set updater
    upd = updater.Updater(optimizer=opt, obj_func=quad, pop_size=10000, threshold=0,
                          out='result', max_iter=10000, logging=True)

    # run IGO and print result
    print(upd.run())

if __name__ == '__main__':
    main()
