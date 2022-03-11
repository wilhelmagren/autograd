"""nn.Module implementations, however, user can define barebone neural nets
by simply creating a callable class that implements a func which extracts
the trainable parameters of the model. Anyway, this file defines the base 
nn.Module object, that acts as skeleton for all default neural net 
operators like Dense, Conv2d, MaxPool2d, BatchNorm2d, etc.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 11-03-2022
License: MIT
"""

from autograd import Tensor
from autograd.ops import Dot


__all__ = [
    Module,
    Dense,
    Conv2d
]


class Module(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError(
        f'user defined nn.Module object has not implemented forward pass')


class Dense(Module):
    def __init__(self, in_size, out_size, initializer='he_uniform'):
        super(Linear, self).__init__()
        self.weight_ = Tensor.uniform(in_size, out_size)

    def forward(self, x):
        return self.weight_.dot(x)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
            padding=(0, 0), initializer='he_uniform'):
        super(Conv2d, self).__init__()
        self.weights_ = [Tensor.uniform(*kernel_size) for _ in range(out_channels)]

    def forward(self, x):
        # x.shape = (B, C, H, W)
        return x
