"""
Implementation of derivable operations.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 16-02-2022
"""
import numpy as np

from .tensor import Function
from .utils import register




class Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / x
register('log', Log)


class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.array([x.sum()])

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * np.ones_like(x)
register('sum', Sum)


class Dot(Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x.dot(w)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = grad_output.dot(w.T)
        grad_w = grad_output.T.dot(x).T
        return (grad_x, grad_w)
register('dot', Dot)



