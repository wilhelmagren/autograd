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
    allops['sub'] = Sub
    allops['sum'] = Sum
    allops['mul'] = Mul
    allops['matmul'] = Matmul
    allops['log'] = Log
    allops['exp'] = Exp
    for name, func in allops.items():
        setattr(Tensor, name, partialmethod(func.apply, func))


class Function(object):
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
        self.requires_grad = any([tensor.requires_grad for tensor in tensors])

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    def apply(self, arg, *x):
        """ arg is the to be initialized context of the function.
        a partialmethod is created
        """
        ctx = arg(self, *x)
        ret = Tensor(ctx.forward(self.data, *[tensor.data for tensor in x]))
        ret._ctx = ctx

        return ret


class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, prev_grad):
        return prev_grad, prev_grad

class Sub(Function):
    def forward(self, x, y):
        return x - y

    def backward(self, prev_grad):
        return prev_grad, -prev_grad

class Sum(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.array([x.sum()])

    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad * np.ones_like(x)

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x * y
    
    def backward(self, prev_grad):
        x, y = self.saved_tensors
        return y*prev_grad, x*prev_grad

class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, prev_grad):
        x, y = self.saved_tensors
        return prev_grad @ y, x.T @ prev_grad

class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.log(x)

    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad / x

class Exp(Function):
    def forward(self, x):
        x = np.exp(x.clip(-80, 80))
        self.save_for_backward(x)
        return x
    
    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad * x



_register_all()
