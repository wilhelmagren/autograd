from autograd import Tensor


class Initializer(object):
    def __call__(self, shape=None):
        if shape is None:
            raise ValueError(
            f'can not initalize tensor when shape is None')

        if not (isinstance(shape, tuple) or isinstance(shape, list)):
            raise ValueError(
            f'provided shape is not a tuple or a list, {shape=}')

        return self.initialize_(shape)

    def initialize_(self, shape):
        raise NotImplementedError(
        f'user defined nn.Initializer object has not implemented initialize_ function')


class Constant(Initializer):
    def __init__(self, val):
        self.val_ = val

    def initialize_(self, shape):
        return Tensor.full(self.val_, *shape, requires_grad=True)

