from autograd import Tensor
from .initializer import Initializer


class Constant(Initializer):
    def __init__(self, val):
        self.val_ = val

    def initialize_(self, shape):
        return Tensor.full(self.val_, *shape, requires_grad=True)

