"""
Implementation of derivable operations.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 16-02-2022
"""
import numpy as np

from .tensor import Function


class Log(Function):
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.log()

    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / x

