# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np


from evoltier.optimizers import CMAES, GaussianNaturalGradientOptimizer
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


def main(gpuID=-1):
    dim = 10
    # set probability distribution
    gaussian = model.MultiVariableGaussian(dim=dim)
    if gpuID >= 0:
        gaussian.use_gpu()

    # set utility function
    w = weight.QuantileBasedWeight(minimization=True, normalize=True)

    # set learning rate of distribution parameters
    #lr = CMAESParameters(dim=dim)
    lr = {'mean': 1/dim, 'cov': 1/(dim**2)}

    # set optimizer
    opt = GaussianNaturalGradientOptimizer(gaussian, w, lr)

    # set updater
    upd = updater.Updater(optimizer=opt, obj_func=quad, pop_size=40, threshold=0,
                          out='result', max_iter=10000, logging=False)

    # run IGO and print result
    print(upd.run())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evoltier Example')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    main(args.gpu)
