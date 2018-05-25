import numpy as np

from ..weight import RankingBasedSelection

class CMASelection(RankingBasedSelection):
    """
    This selection scheme is Non-increasing transformation as CMA-ES weight. See also,
    [Hansen & Auger, 2014]<https://arxiv.org/abs/1604.00772>
    """

    def transform(self, ranking, xp=np):
        lam = len(ranking)
        weight = xp.maximum(0, xp.log((lam + 1) / 2) - xp.log(ranking))
        weight /= weight.sum()
        return weight
