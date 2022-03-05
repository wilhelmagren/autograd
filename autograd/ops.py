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
                        

class Add(Function):
    def forward(self, x, y):
        return x + y
    
    def backward(self, prev_grad):
        return prev_grad, prev_grad


class Dot(Function):
    def forward(self, x, w):
        self.save_for_backward(x, w)
        return x.dot(w)
    
    def backward(self, prev_grad):
        x, w, = self.saved_tensors
        return prev_grad.dot(w.T), x.T.dot(prev_grad)
    
    
setattr(Tensor, 'add', partialmethod(Add.apply, Add))
setattr(Tensor, 'dot', partialmethod(Dot.apply, Dot))
