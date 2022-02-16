"""
Implementation of Tensor and Function objects.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 16-02-2022
"""
import numpy as np


class Tensor(object):
    def __init__(self, data, device='cpu', requires_grad=False):
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                    'Error constructing tensor with {data=}')

        self.data = data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)
    
    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

    @classmethod
    def arange(cls, stop, start=0, **kwargs):
        return cls(np.arange(stop=stop, start=start).astype(np.float32), **kwargs)

    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls((np.random.uniform(-1., 1., size=shape)
            /np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)

    


class Function(object):
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
        self.requires_grad = any([tensor.requires_grad for tensor in tensors])

    def save_for_backward(self, *x):
        if self.requires_grad:
            self.saved_tensors.extend(*x)




