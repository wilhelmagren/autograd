import numpy as np

from autograd import Tensor
from .initalizer import Initializer, get_fans


class XavierUniform(Initializer):
    """ fills the input Tensor with values according to the method
    described in `Understanding the difficulty to train deep
    feedforward neural networks` - Glorot, X. & Bengio, Y.(2010),
    using a uniform distribution. The resulting Tensor will have
    values sampled from U(-a, a) where a is calculated based on 
    a gain parameter and the fan constants of the Tensor.
    """
    def initialize_(self, shape, gain=1.0):
        fan_in, fan_out = get_fans(shape)
        alpha = gain * np.sqrt(6/(fan_in + fan_out))
        return Tensor(np.uniform(-alpha, alpha, size=shape))


class XavierNormal(Initializer):
    """ fills the input Tensor with values according to same methods
    as mentioned above for XaviverUniform, Glorot, X. & Bengio, Y.(2010),
    but using a normal distribution instead. Thus, the resulting Tensor
    will have values sampled from N(0, std^2) where std is calculated based
    on a gain parameter and the fan constants of the Tensor.
    """
    def initialize_(self, shape, gain=1.0):
        fan_in, fan_out = get_fans(shape)
        std = gain * np.sqrt(2/(fan_in + fan_out))
        return Tensor(np.random.normal(scale=std, size=shape))

