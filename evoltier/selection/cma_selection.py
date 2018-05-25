import numpy as np

from ..weight import RankingBasedSelection

class CMASelection(RankingBasedSelection):
    """
    This selection scheme is Non-increasing transformation as CMA-ES weight. See also,
    [Hansen & Auger, 2014]<https://arxiv.org/abs/1604.00772>
    """

    def transform(self, rank_based_vals, xp=np):
        lam = len(rank_based_vals)
        weight = xp.maximum(0, xp.log((lam + 1) / 2) - xp.log(rank_based_vals))
        weight /= weight.sum()
        return weight
