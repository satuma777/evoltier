# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np


def cma_like_weight(q_plus, xp):
    """
    Non-increasing function of CMA-ES weight which is
    considered for large population size. See also,
    [Shirakawa et al. 2015(GECCO2015)]<http://shiralab.ynu.ac.jp/data/paper/gecco2015_shirakawa.pdf>
    """
    
    weight_plus = -2. * xp.log(2. * q_plus)
    weight_plus[q_plus > 0.5] = 0

    return weight_plus


class QuantileBasedWeight(object):
    def __init__(self, minimization=True, non_increasing_function=cma_like_weight, normalize=False):
        self.min = minimization
        self.non_inc_func = non_increasing_function
        self.normalize = normalize

    def __call__(self, evaluation, likelihood_ratio=None, xp=np):
        pop_size = evaluation.shape[0]
        if likelihood_ratio is None:
            likelihood_ratio = xp.ones(pop_size)
        q_plus = self.compute_quantile(evaluation, likelihood_ratio, pop_size, xp=xp)
        weight = self.non_inc_func(q_plus, xp) / pop_size
        if self.normalize:
            return weight / xp.linalg.norm(weight, ord=1)
        else:
            return weight

    def compute_quantile(self, evaluation, likelihood_ratio, pop_size, xp=np, rank_rule='upper'):
        sorter = xp.argsort(evaluation)
        if not self.min:
            sorter = sorter[::-1]

        # set label sequentially that minimum eval =  0 , ... , maximum eval = pop_size - 1
        # --- Example ---
        # eval = [12, 13, 10]
        #  inv = [ 1,  2,  0]
        inv = xp.empty(sorter.size, dtype=xp.integer)
        inv[sorter] = xp.arange(sorter.size, dtype=xp.integer)

        arr = evaluation[sorter]
        obs = xp.r_[True, arr[1:] != arr[:-1]]
        dense = xp.cumsum(obs)[inv]

        # cumulative counts of likelihood ratio
        count = xp.r_[False, xp.cumsum(likelihood_ratio[sorter])]

        if rank_rule == 'upper':
            cum_llr = count[dense]
        elif rank_rule == 'lower':
            cum_llr = count[dense - 1]

        quantile = cum_llr / pop_size
        return quantile

#TODO: LebesgueMeasureBasedWeight [Akimoto2012(GECCO2012)]