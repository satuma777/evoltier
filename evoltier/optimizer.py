class Optimizer(object):
    """
    Base class of all optimizer.
    """
    
    def __init__(self, distribution, weight_function, lr):
        self.target = distribution
        self.t = 0
        self.w_func = weight_function
        self.lr = lr
        
        if not callable(self.w_func):
            raise TypeError('weight function is NOT callable.')
    
    def update(self):
        raise NotImplementedError()
    
    def get_info_dict(self):
        raise NotImplementedError()
    
    def generate_header(self):
        raise NotImplementedError()
