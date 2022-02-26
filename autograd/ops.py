import numpy as np

from functools import partialmethod
from .tensor import Tensor


__all__ = [
    'add',
    'sub',
    'mul',
    'dot',
    'matmul',
    'sum'
]


def _register_all():
    allops = {}
    allops['add'] = Add
    for name, func in allops.items():
        setattr(Tensor, name, partialmethod(func.apply, func))


class Function(object):
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
        self.requires_grad = any([tensor.requires_grad for tensor in tensors])

    def save_for_backward(self, *x):
        self.saved_tensors.extend(*x)

    def apply(self, arg, *x):
        """ arg is the to be initialized context of the function.
        a partialmethod is created
        """
        ctx = arg(self, *x)
        ret = Tensor(arg.forward(self.data, *[tensor.data for tensor in x]))
        ret._ctx = ctx

        return ret


class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, prev_grad):
        return prev_grad, prev_grad


_register_all()
