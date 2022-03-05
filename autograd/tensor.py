import numpy as np


class Tensor(object):
    def __init__(self, data, requires_grad=True, dtype=np.float32):
        self.requires_grad = requires_grad
        
        if isinstance(data, list):
            data = np.array(data).astype(dtype)
        elif isinstance(data, np.ndarray):
            data = data.astype(dtype)
        elif isinstance(data, np.float32) or isinstance(data, np.uint8):
            data = np.array([data]).astype(data.dtype)
        else:
            raise ValueError(
                f'unknown data instance passed to Tensor.__init__, {type(data)}')
        
        self.data = data
        self.grad = None
        self._ctx = None
    
    def __repr__(self):
        return f''
    
    def __str__(self):
        return f'<autograd.Tensor\n{self.data}\n' \
        f'dtype={self.dtype}, grad_fn={self._ctx}, grad={self.grad}>'
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def uint8(self):
        self.data = self.data.astype(np.uint8)
        return self
    
    def float32(self):
        self.data = self.data.astype(np.float32)
        return self
    
    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape), **kwargs)
    
    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape), **kwargs)
    
    @classmethod
    def eye(cls, dims, **kwargs):
        return cls(np.eye(dims), **kwargs)
    
    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls(np.random.uniform(-1.0, 1.0, size=shape) 
        / np.sqrt(np.prod(shape)), **kwargs)
    
    @classmethod
    def full(cls, val, *shape, **kwargs):
        return cls(np.full(shape, val), **kwargs)
    
    