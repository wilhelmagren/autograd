"""
Implementation of Tensor and Function objects.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 16-02-2022
"""
import numpy as np

from functools import partialmethod
from .device import _get_device



class Tensor(object):
    def __init__(self, data, device='cpu', requires_grad=False):
        self.device = _get_device(device)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.float32):
            data = np.array(data, dtype=np.float32)

        self.data = data

    def __repr__(self):
        return f'<autograd.tensor.Tensor:\n{self.data} device={self.device} _ctx={self._ctx} grad={self.grad}>'
    
    def backward(self, allow_fill=True):
        """func performs backward pass in the created
        directed acyclic graph. 

        Parameters
        ----------
        allow_fill: bool
            First gradient calculation is nothing, so assign it 1
            since we are multiplying the output gradient with the 
            new gradient. Otherwise, all becomes zero.

        """
        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        gradients = self._ctx.backward(self._ctx, self.grad)
        gradients = [gradients] if len(self._ctx.parents) == 1 else gradients

        for tensor, gradient in zip(self._ctx.parents, gradients):
            if gradient is None:
                continue

            tensor.grad = gradient
            tensor.backward(allow_fill=False)
        
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

    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div)



class Function(object):
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
        self.requires_grad = any([tensor.requires_grad for tensor in tensors])

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    def apply(self, arg, *x):
        """
        First, create the function which is the attribute arg,
        the tensors *x are saved as parents attributes in the 
        context ctx. Perform the forward pass on the created
        function, applying the operation between the contexts
        data and the argument data in *x.
        Save the function context in the return tensor.
        """
        ctx = arg(self, *x)
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx
        return ret
