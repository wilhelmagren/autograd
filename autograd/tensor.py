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

        assert(self.grad is not None)

        gradients = self._ctx.backward(self._ctx, self.grad)
        gradients = [gradients] if len(self._ctx.parents) == 1 else gradients
        for t, g in zip(self._ctx.parents, gradients):
            if g.shape != t.data.shape:
                raise ValueError(
                        f'Gradient shape must match tensor shape, {g.shape} {t.data.shape}')
            t.grad = g
            t.backward(allow_fill=False)

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
        self.saved_tensors.extend(x)

    def apply(self, arg, *x):
        ctx = arg(self, *x)
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx
        return ret


def register(name, func):
    setattr(Tensor, name, partialmethod(func.apply, func))


