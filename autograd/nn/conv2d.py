from autograd import Tensor
from autograd.nn import Module


class Conv2d(Module):
    def __init__(self, out_channels, in_channels, height, width, stride=1, padding=0):
        if isinstance(stride, int):
            stride = (stride, stride)
        
        if isinstance(padding, int):
            padding = (padding, padding)

        for s in stride: assert s >= 1
        for p in padding: assert p>= 0

        self.in_channels_ = in_channels
        self.height_ = height
        self.width_ = width
        self.stride_ = stride
        self.padding_ = padding
        self.kernels_ = Tensor.uniform(out_channels, in_channels, height, width)

    def forward(self, x):
        return x.conv2d(self.kernels_)

