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
    
    def __call__(self, evaluation, factor=None):
        xp = self.xp
        pop_size = evaluation.shape[0]
        q_plus = None
        
        if factor is None:
            factor = xp.ones_like(evaluation)
        
        if self.min:
            q_plus = xp.array([factor[(evaluation <= e)].sum() for e in evaluation])
            q_plus /= pop_size
        else:
            q_plus = xp.array([factor[(evaluation >= e)].sum() for e in evaluation])
            q_plus /= pop_size
        
        return self.non_inc_func(q_plus, self.xp) / pop_size

#TODO: LebesgueMeasureBasedWeight [Akimoto2012(GECCO2012)]