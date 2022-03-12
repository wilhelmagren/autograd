from autograd import Tensor


class Initializer(object):
    def __call__(self, shape=None):
        if shape is None:
            raise ValueError(
            f'can not initialize Tensor when shape is None')

        if not isinstance(shape, tuple):
            raise ValueError(
            f'provided shape is not a tuple, {shape=}')

        return self.initialize_(shape)

    def initialize_(self, shape):
        raise NotImplementedError(
        f'user defined Initializer has not implemented initialize_ func')
