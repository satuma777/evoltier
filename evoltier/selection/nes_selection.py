import numpy as np

from ..weight import RankingBasedSelection

class NESSelection(RankingBasedSelection):
    """
    This selection scheme is Non-increasing transformation as NES weight. See also,
    [Wierstra et. al., 2014]<http://jmlr.org/papers/v15/wierstra14a.html>
    """

    def transform(self, ranking, xp=np):
        lam = len(ranking)
        weight = xp.maximum(0, xp.log((lam / 2) + 1) - xp.log(ranking))
        weight /= weight.sum()
        return weight - 1. / lam
