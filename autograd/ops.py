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
        output = Tensor(ctx.forward(self.data, *[t.data for t in inputs], **kwargs))
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
        

class Exp(Function):
    def forward(self, x):
        result = np.exp(x)
        self.save_for_backward(result)
        return result

    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad * x


class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.log(x)
    
    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad / x


class Sigmoid(Function):
    def forward(self, x):
        result = 1 / (1 + np.exp(-x))
        self.save_for_backward(result)
        return result
    
    def backward(self, prev_grad):
        x, = self.saved_tensors
        return prev_grad * x * (1 - x)


class Conv2d(Function):
    def forward(self, x, k, stride=1):
        if isinstance(stride, int):
            stride = (stride, stride)
        
        batch_size, input_C_, input_H, input_W = x.shape
        output_C, input_C, kernel_H, kernel_W = k.shape

        assert input_C_ == input_C
        
        self.save_for_backward(x, k)
        stride_H, stride_W = stride
        out_H = 1 + (input_H - kernel_H) // stride_H
        out_W = 1 + (input_W - kernel_W) // stride_W
        mat_k = k.reshape(output_C, -1).T

        featuremaps = np.zeros((batch_size, output_C, out_H,
            out_W)).astype(x.dtype)
        for h in range(out_H):
            for w in range(out_W):
                mat_x = x[:, :, h:h+kernel_H, w:w+kernel_W].reshape(batch_size, -1)
                featuremaps[:, :, h, w] = mat_x.dot(mat_k)

        return featuremaps

    def backward(self, prev_grad):
        batch_size, _, out_H, out_W = prev_grad.shape
        x, k = self.saved_tensors
        output_C, input_C, kernel_H, kernel_W = k.shape
        mat_k = k.reshape(output_C, -1)

        dx, dk = np.zeros_like(x), np.zeros_like(k)
        for h in range(out_H):
            for w in range(out_W):
                g = prev_grad[:, :, h, w]
                mat_x = x[:, :, h:h+kernel_H, w:w+kernel_W].reshape(batch_size, -1)
                dk += g.T.dot(mat_x).reshape(dk.shape)
                dx[:, :, h:h+kernel_H, w:w+kernel_W] += g.dot(mat_k).reshape(batch_size,
                        input_C, kernel_H, kernel_W)

        return dx, dk


__allops__ = [
    ('mean', Mean),
    ('sum', Sum),
    ('add', Add),
    ('sub', Sub),
    ('dot', Dot),
    ('mul', Mul),
    ('relu', ReLU),
    ('logsoftmax', LogSoftmax),
    ('exp', Exp),
    ('log', Log),
    ('sigmoid', Sigmoid),
    ('conv2d', Conv2d)
]


for name, func in __allops__:
    setattr(Tensor, name, partialmethod(func.apply, func))

