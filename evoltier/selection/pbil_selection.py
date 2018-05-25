import numpy as np
from math import floor

from ..weight import RankingBasedSelection


class PBILSelection(RankingBasedSelection):
    """
    This selection scheme is used by PBIL and compact GA.
    Also, PBIL selection scheme also used in natural gradient update
    with Bernoulli distribution. See also,
    [Shirakawa et al. 2018 (AAAI-2018)]<https://arxiv.org/abs/1801.07650>
    """
    def __init__(self, selection_rate=0.5, is_use_negative=True, is_minimize=True, is_normalize=False):
        super(PBILSelection, self).__init__(is_minimize, is_normalize)
        self.selection_rate = selection_rate
        self.is_use_negative = is_use_negative

    def transform(self, rank_based_vals, xp=np):
        weights = xp.zeros_like(rank_based_vals)
        worst_rank = len(rank_based_vals)
        idx_sorted_rank = xp.argsort(rank_based_vals)

        if self.is_use_negative:
            half_num_weight = floor(worst_rank * self.selection_rate / 2.)
            # the best floor(lam * selection_rate / 2) samples get the positive weights
            idx_positive = idx_sorted_rank[:half_num_weight]
            weights[idx_positive] = 1
            # the worst floor(lam * selection_rate / 2) samples get the negative weights
            idx_negative = idx_sorted_rank[-half_num_weight:]
            weights[idx_negative] = -1
        else:
            # the best floor(lam * selection_rate) samples get the positive weights
            num_weight = floor(worst_rank * self.selection_rate)
            idx_positive = idx_sorted_rank[:num_weight]
            weights[idx_positive] = 1

        return weights
