"""
Implementation of derivable operations.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 17-02-2022
"""
import numpy as np

from .tensor import Tensor
from .tensor import Function
from functools import partialmethod


__allops__ = [
    'add',
    'sub',
    'mul',
    'dot',
    'div',
    'pow',
    'log',
    'sum',
    'logsoftmax',
    'relu'
]

"""
__allfuncs__ = [
    Add,
    Sub,
    Mul,
    Dot,
    Div,
    Pow,
    Log,
    Sum
]
"""

def _register(names, funcs):
    """

    partialmethod freezes the given method with the specific
    argument. So the method func.apply is frozen with 
    the function as argument. This decides how func.apply has
    to be implemented, since the argument is the function 
    to apply...

    Parameters
    ----------
    ops: list | tuple
        Iterable collection of operations to register.
        Should be either __allops__ or a subset of them.
    """
    for name, func in zip(names, funcs):
        if name in __allops__:
            setattr(Tensor, name, partialmethod(func.apply, func))



class Dot(Function):
    """
    Gradient
    --------
    Given vectors u and v, the gradient of the dot product becomes
    grad(u * v) = grad(u).T*v + grad(v).T*u
    """
    
    def __str__(self):
        return 'Dot'

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x.dot(w)

    @staticmethod
    def backward(ctx, prev_grad):
        x, w = ctx.saved_tensors
        grad_x = prev_grad.dot(w.T)
        grad_w = prev_grad.T.dot(x).T
        return (grad_x, grad_w)


class Add(Function):

    def __str__(self):
        return '<autograd.ops.Add>'

    @staticmethod
    def forward(ctx, x, y):
        return x + y
    
    @staticmethod
    def backward(ctx, prev_grad):
        return (prev_grad, prev_grad)


class Mul(Function):

    def __str__(self):
        return '<autograd.ops.Mul>'

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x * w
    
    @staticmethod
    def backward(ctx, prev_grad):
        x, w = ctx.saved_tensors
        return (w * prev_grad, x * prev_grad)


class Sum(Function):

    def __str__(self):
        return '<autograd.ops.Sum>'

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.array([x.sum()])
    
    @staticmethod
    def backward(ctx, prev_grad):
        x, = ctx.saved_tensors
        return prev_grad * np.ones_like(x)


class ReLU(Function):
    
    def __str__(self):
        return '<autograd.ops.ReLU>'

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(x, 0)
    
    @staticmethod
    def backward(ctx, prev_grad):
        x, = ctx.saved_tensors
        grad = prev_grad.copy()
        grad[x < 0] = 0
        return grad


class LogSoftmax(Function):

    def __str__(self):
        return '<autograd.ops.LogSoftmax>'

    @staticmethod
    def forward(ctx, x):
        def logsumexp(y):
            c = y.max(axis=1)
            return c + np.log(np.exp(y - c.reshape((-1, 1))).sum(axis=1))
        ls = x - logsumexp(x).reshape((-1, 1))
        ctx.save_for_backward(ls)
        return ls

    @staticmethod
    def backward(ctx, prev_grad):
        x, = ctx.saved_tensors
        return x - np.exp(x) * x.sum(axis=1).reshape((-1, 1))

_register(['dot', 'add', 'mul', 'sum', 'relu', 'logsoftmax'], [Dot, Add, Mul, Sum, ReLU, LogSoftmax])
