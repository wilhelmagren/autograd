from autograd import Tensor
from autograd.nn import Module


class LogSoftmax(Module):
    def __init__(self, axis=None):
        self.axis_ = axis

    def forward(self, x):
        return x.logsoftmax()

