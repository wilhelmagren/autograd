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
        elif isinstance(data, np.float64):
            data = np.array([data]).astype(np.float32)
        elif isinstance(datam np.uint16):
            data = np.array([data]).astype(np.uint8)
        else:
            raise ValueError(
                f'unknown data instance passed to Tensor.__init__, {type(data)}')
        
        self.data = data
        self.grad = None
        self._ctx = None
    
    def __repr__(self):
        return f'<{self.__class__.__module__}.{self.__class__.__qualname__}\n' \
        f'{self.data}>'
    
    def __str__(self):
        return f'<autograd.Tensor\n{self.data}\n' \
        f'dtype={self.dtype}, grad_fn={self._ctx}, grad={self.grad}>'
    
    def uint8(self):
        self.data = self.data.astype(np.uint8)
        return self
    
    def float32(self):
        self.data = self.data.astype(np.float32)
        return self
     
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
   
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
    
    def backward(self, allow_fill=True):
        if self._ctx is None:
            return
            
        if self.grad is None and allow_fill:
            assert self.shape == (1, )
            self.grad = np.ones_like(self.data).reshape((-1, 1))
        
        parents = self._ctx.parents
        gradients = self._ctx.backward(self.grad)
        gradients = [gradients] if len(parents) == 1 else gradients
        
        for gradient, parent in zip(gradients, parents):
            if gradient is None:
                continue
            
            if parent.requires_grad:
                parent.grad = gradient
            
            parent.backward(allow_fill=False)
        
