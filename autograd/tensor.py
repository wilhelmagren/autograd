import numpy as np
from functools import partialmethod

class Tensor(object):
    def __init__(self, data, requires_grad=True):
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

        if isinstance(data, list) or isinstance(data, np.float32):
            data = np.array(data, dtype=np.float32)
        else:
            raise ValueError(
                    f'Unknown data given to Tensor, {data}')

        self.data = data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype
