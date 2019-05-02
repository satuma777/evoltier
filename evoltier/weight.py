from __future__ import division
import numpy as np


class RankingBasedSelection(object):
    def __init__(self, is_minimize=True, is_normalize=False):
        self.is_minimize = is_minimize
        self.is_normalize = is_normalize

    def __call__(self, evals, coefficient=None, xp=np):
        ranking = self.compute_ranking(evals, coefficient=None, xp=xp)
        weight = self.transform(ranking, xp=xp)
        if self.is_normalize:
            weight /= xp.linalg.norm(weight, ord=1)
        return weight

    def compute_ranking(self, evals, coefficient=None, xp=np, rank_rule='upper'):
        pop_size = evals.shape[0]
        if coefficient is None:
            coefficient = xp.ones(pop_size)
        sorter = xp.argsort(evals)
        if self.is_minimize is False:
            sorter = sorter[::-1]

        # set label sequentially that minimum eval =  0 , ... , maximum eval = pop_size - 1
        # --- Example ---
        # eval = [12, 13, 10]
        #  inv = [ 1,  2,  0]
        inv = xp.empty(sorter.size, dtype=xp.integer)
        inv[sorter] = xp.arange(sorter.size, dtype=xp.integer)

        arr = evals[sorter]
        obs = xp.r_[True, arr[1:] != arr[:-1]]
        dense = xp.cumsum(obs)[inv]

        # cumulative counts of likelihood ratio
        count = xp.r_[False, xp.cumsum(coefficient[sorter])]

        ranking = count[dense] if rank_rule == 'upper' else count[dense - 1]
        return ranking

    def transform(self, rank_based_vals, xp=np):
        raise NotImplementedError()


class QuantileBasedSelection(RankingBasedSelection):
    def __init__(self, is_minimize=True, is_normalize=False):
        super(QuantileBasedSelection, self).__init__(is_minimize, is_normalize)

    def __call__(self, evals, coefficient=None, xp=np,):
        quntiles = self.compute_ranking(evals, coefficient=coefficient, xp=xp) / len(evals)
        weight = self.transform(quntiles, xp=xp)
        if self.is_minimize:
            weight /= xp.linalg.norm(weight, ord=1)
        return weight

    def transform(self, rank_based_vals, xp=np):
        raise NotImplementedError()