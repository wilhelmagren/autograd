from autograd import Tensor
from autograd.nn import Module


class ReLU(Module):
    def __init__(self, inplace=False):
        self.inplace_ = inplace

    def forward(self, x):
        if self.inplace_:
            raise NotImplementedError(
            f'still have not figured out how to do this with nn.Sequential model')

        return x.relu()


class LogSoftmax(Module):
    def __init__(self, axis=None):
        self.axis_ = axis

    def forward(self, x):
        return x.logsoftmax()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

