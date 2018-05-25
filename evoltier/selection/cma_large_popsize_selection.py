import numpy as np

from ..weight import QuantileBasedSelection

class CMALargePopSizeSelection(QuantileBasedSelection):
    """
    This selection scheme is Non-increasing transformation as CMA-ES weight which is
    considered for large population size. See also,
    [Shirakawa et al. 2015 (GECCO2015)]<http://shiralab.ynu.ac.jp/data/paper/gecco2015_shirakawa.pdf>
    """

    def transform(self, quantiles, xp=np):
        weight = -2. * xp.log(2. * quantiles)
        weight[quantiles > 0.5] = 0
        return weight
