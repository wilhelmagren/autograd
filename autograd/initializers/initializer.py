from autograd import Tensor


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return (fan_in, fan_out)


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

