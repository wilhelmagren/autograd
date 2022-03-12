from autograd import Tensor
from autograd.nn import Module

class Dense(Module):
    def __init__(self, in_shape, out_shape):
        self.weight_ = Tensor.uniform(in_shape, out_shape)

    def forward(self, x):
        return x.dot(self.weight_)

