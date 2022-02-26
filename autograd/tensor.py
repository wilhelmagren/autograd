import numpy as np

class Tensor(object):
    def __init__(self, data, requires_grad=True):
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

        if isinstance(data, list) or isinstance(data, np.float32):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        else:
            raise ValueError(
                    f'Unknown data given to Tensor, {data}')

        self.data = data

    def __repr__(self):
        return f'<autograd.tensor.Tensor {self.data}, grad={self.grad}, _ctx={self._ctx}>'

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls(np.random.uniform(-1., 1., size=shape)
                / np.sqrt(np.prod(shape)).astype(np.float32), **kwargs)

    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)
    
    def mean(self):
        div = Tensor(np.array([1 / self.data.size], dtype=np.float32))
        return self.sum().mul(div)

    def backward(self, allow_fill=True):
        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            self.grad = np.ones_like(self.data)
        
        parents = self._ctx.parents
        gradients = self._ctx.backward(self.grad)
        gradients = [gradients] if len(parents) == 1 else gradients
        
        for tensor, gradient in zip(parents, gradients):
            if gradient is None:
                continue
            
            tensor.grad = gradient
            tensor.backward(allow_fill=False)

