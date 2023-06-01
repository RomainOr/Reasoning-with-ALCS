import numpy as np

class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance

class RandomNumberGenerator(Singleton):

    rng = np.random.default_rng()

    @classmethod
    def seed(cls, seed):
        cls.rng = np.random.default_rng(seed)

    @classmethod
    def random(cls):
        return cls.rng.random()
    
    @classmethod
    def choice(cls, a, size=None, replace=True, p=None):
        return cls.rng.choice(a, size=size, replace=replace, p=p)
    
    @classmethod
    def integers(cls, low, high=None, size=None, dtype=np.int64, endpoint=False):
        return cls.rng.integers(low, high=high, size=size, dtype=dtype, endpoint=endpoint)
    
    @classmethod
    def uniform(cls, low=0.0, high=1.0, size=None):
        return cls.rng.uniform(low=low, high=high, size=size)
