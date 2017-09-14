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
    weight_plus[xp.where(q_plus > 0.5)] = 0

    return weight_plus


class QuantileBasedWeight(object):
    def __init__(self, minimization=True, non_increasing_function=cma_like_weight, xp=np):
        self.min = minimization
        self.non_inc_func = non_increasing_function
        self.xp = xp

    def __call__(self, evaluation, likelihood_ratio=None):
        pop_size = evaluation.shape[0]
        if likelihood_ratio is None:
            likelihood_ratio = self.xp.ones(pop_size)
        
        if self.min:
             q_plus = self.compute_quantile(evaluation, likelihood_ratio, pop_size)
        else:
             q_plus = self.compute_quantile(evaluation, likelihood_ratio, pop_size)
        
        weight = self.non_inc_func(q_plus, self.xp) / pop_size
        return weight
    
    def compute_quantile(self, evaluation, likelihood_ratio, pop_size, type='upper'):
        # inspired by scipy.stats.rankdata
        
        xp = self.xp
        flatten_array = xp.ravel(np.asarray(evaluation))
        sorter = np.argsort(flatten_array)
    
        # set label sequentially that minimum eval =  0 , ... , maximum eval = pop_size - 1
        # --- Example ---
        # eval = [12, 13, 10]
        #  inv = [ 1,  2,  0]
        inv = np.empty(sorter.size, dtype=xp.intp)
        inv[sorter] = np.arange(sorter.size, dtype=xp.intp)
    
        arr = evaluation[sorter]
        obs = xp.r_[True, arr[1:] != arr[:-1]]
        dense = obs.cumsum()[inv]
        if not self.min:
            dence = - (dense - pop_size) + 1
    
        # cumulative counts of likelihood ratio
        count = xp.r_[False, likelihood_ratio[sorter].cumsum()]
    
        if type == 'upper':
            quantile = count[dense]
        elif type == 'lower':
            quantile = (count[dense] - 1)
            
        return quantile / pop_size

#TODO: LebesgueMeasureBasedWeight [Akimoto2012(GECCO2012)]