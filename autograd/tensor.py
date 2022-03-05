import numpy as np


class Tensor(object):
    def __init__(self, data, requires_grad=True, dtype=np.float32):
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
        
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
    