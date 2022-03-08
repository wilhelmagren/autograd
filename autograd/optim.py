import numpy as np


class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = [param for param in parameters if param.requires_grad]

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    def step(self):
        raise NotImplementedError(
        f'user defined Optimizer object has not implemented gradient update')


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-3):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros(t.shape) for t in self.parameters]
        self.v = [np.zeros(t.shape) for t in self.parameters]

    def step(self):
        self.t = self.t + 1
        alpha = self.lr * ((1 - self.beta2 ** self.t) ** 0.5) / (1 - self.beta1 ** self.t)
        for i, param in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

            param.data -= alpha * self.m[i] / ((self.v[i]) ** 0.5 + self.eps)

