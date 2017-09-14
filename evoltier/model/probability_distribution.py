import numpy as np


class ProbabilityDistribution(object):
    def __init__(self, xp=np):
        self.xp = xp
    
    def sampling(self):
        raise NotImplementedError()
    
    def get_param(self):
        raise NotImplementedError()

    def set_param(self):
        raise NotImplementedError()

    def calculate_log_likelihood(self):
        raise NotImplementedError()
    
    def get_info(self):
        raise NotImplementedError()
    
    def get_info_dict(self):
        raise NotImplementedError()
    
    def generate_header(self):
        raise NotImplementedError()

    def use_gpu(self):
        try:
            import cupy as cp
            self.xp = cp
        except ImportError:
            raise ImportError()
