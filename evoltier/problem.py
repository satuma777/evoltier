
class Problem(object):
    """
    Abstact class of a target problem or a objective function.
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

