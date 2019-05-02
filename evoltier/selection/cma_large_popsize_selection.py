import numpy as np

from ..weight import QuantileBasedSelection


class CMALargePopSizeSelection(QuantileBasedSelection):
    """
    This selection scheme is Non-increasing transformation as CMA-ES weight which is
    considered for large population size. See also,
    [Shirakawa et al. 2015 (GECCO2015)]<http://shiralab.ynu.ac.jp/data/paper/gecco2015_shirakawa.pdf>
    """

    def transform(self, rank_based_vals, xp=np):
        weight = -2. * xp.log(2. * rank_based_vals)
        weight[rank_based_vals > 0.5] = 0
        return weight
