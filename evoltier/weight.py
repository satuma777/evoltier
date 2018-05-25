from __future__ import division
import numpy as np


class QuantileBasedSelection(object):
    def __init__(self, is_minimize=True, is_normalize=False):
        self.is_minimize = is_minimize
        self.is_normalize = is_normalize

    def __call__(self, evals, coefficient=None, xp=np):
        quantiles = self.compute_quantiles(evals, coefficient=None, xp=xp)
        weight = self.transform(quantiles, xp=xp)
        if self.is_normalize:
            weight /= xp.linalg.norm(weight, ord=1)
        return weight

    def compute_quantiles(self, evals, coefficient=None, xp=np, rank_rule='upper'):
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

        if rank_rule == 'upper':
            cum_llr = count[dense]
        elif rank_rule == 'lower':
            cum_llr = count[dense - 1]

        quantile = cum_llr / pop_size
        return quantile

    def transform(self, rank_based_vals, xp=np):
        raise NotImplementedError()


class RankingBasedSelection(QuantileBasedSelection):
    def __init__(self, is_minimize=True, is_normalize=False):
        super(RankingBasedSelection, self).__init__(is_minimize, is_normalize)

    def __call__(self, evals, coefficient=None, xp=np,):
        ranking = self.compute_ranking(evals, coefficient=coefficient, xp=xp)
        weight = self.transform(ranking, xp=xp)
        if self.is_minimize:
            weight /= xp.linalg.norm(weight, ord=1)
        return weight

    def compute_ranking(self, evals, coefficient=None, xp=np):
        return self.compute_quantiles(evals, coefficient=coefficient, xp=xp) * len(evals)
