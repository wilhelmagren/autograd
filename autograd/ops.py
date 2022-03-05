import numpy as np

from functools import partialmethod
from autograd import Tensor



class Function(object):
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
        self.requires_grad = any([t.requires_grad for t in tensors])
    
    def __str__(self):
        return f'<autograd.ops.{self.__class__.__qualname__}>'
    
    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)
    
    def apply(self, op, *inputs, **kwargs):
        ctx = op(self, *inputs)
        output = Tensor(ctx.forward(self.data, *[t.data for t in inputs]), **kwargs)
        output._ctx = ctx
        return output

    
class Mean(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.sum() / np.prod(x.shape)
    
    def backward(self, prev_grad):
        x, = self.saved_tensors
        return np.ones_like(x) * prev_grad / np.prod(x.shape)

    
class Sum(Function):
    def forward(self, x, axis=None, keepdims=True):
        self.save_for_backward(x)
        return x.sum(axis=axis, keepdims=keepdims)
    
    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad * np.ones_like(x)


class Add(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x + y
    
    def backward(self, prev_grad):
        x, y, = self.saved_tensors
        return prev_grad, prev_grad


class Sub(Function):
    def forward(self, x, y):
        return x - y
    
    def backward(self, prev_grad):
        return prev_grad, -prev_grad


class Dot(Function):
    def forward(self, x, w):
        self.save_for_backward(x, w)
        return x.dot(w)
    
    def backward(self, prev_grad):
        x, w, = self.saved_tensors
        return prev_grad.dot(w.T), x.T.dot(prev_grad)


class Mul(Function):
    def forward(self, x, w):
        self.save_for_backward(x, w)
        return x * w
    
    def backward(self, prev_grad):
        x, w = self.saved_tensors
        return prev_grad * w, prev_grad * x


class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.maximum(x, 0)
    
    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad * (x >= 0)


class LogSoftmax(Function):
    def forward(self, x):
        def logsumexp_(z):
            m = z.max(axis=1)
            return m + np.log(np.exp(z - m.reshape((-1, 1))).sum(axis=1))
        logsumexp = x - logsumexp_(x).reshape((-1, 1))
        self.save_for_backward(logsumexp)
        return logsumexp
    
    def backward(self, prev_grad):
        logsumexp, = self.saved_tensors
        return prev_grad - np.exp(logsumexp) * prev_grad.sum(axis=1).reshape((-1, 1))
        

__allops__ = [
    ('mean', Mean),
    ('sum', Sum),
    ('add', Add),
    ('sub', Sub),
    ('dot', Dot),
    ('mul', Mul),
    ('relu', ReLU),
    ('logsoftmax', LogSoftmax)
]


for name, func in __allops__:
    setattr(Tensor, name, partialmethod(func.apply, func))

