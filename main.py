# -*- coding: utf-8 -*-
from __future__ import print_function

from evoltier.optimizers import CMAES, GaussianNaturalGradientOptimizer, BernoulliNaturalGradientOptimizer
from evoltier import updater
from evoltier import weight
from evoltier import model
from evoltier.utils import CMAESParameters


def quad(x):
    # implementation of quadratic function.
    # global minima is zero.
    return (x * x).sum(axis=1)


def negative_quad(x):
    # implementation of quadratic function.
    # global mixima is zero.
    return - (x * x).sum(axis=1)

def onemax(x):
    return x.sum(axis=1)

def leading_ones(x):
    import numpy as np
    return np.sum(np.dot(x, np.tri(x.shape[1])), axis=1)

def main(gpuID=-1):
    dim = 100
    # set probability distribution
    gaussian = model.Bernoulli(dim=dim)
    if gpuID >= 0:
        gaussian.use_gpu()

    # set utility function
    w = weight.QuantileBasedWeight(minimization=False, normalize=True)

    # set learning rate of distribution parameters
    #lr = CMAESParameters(dim=dim)
    lr = {'theta': 1 / dim}

    # set optimizer
    opt = BernoulliNaturalGradientOptimizer(gaussian, w, lr)

    # set updater
    upd = updater.Updater(optimizer=opt, obj_func=leading_ones, pop_size=40, threshold=10000,
                          out='result', max_iter=10000, logging=True)

    # run IGO and print result
    print(upd.run())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evoltier Example')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    main(args.gpu)
