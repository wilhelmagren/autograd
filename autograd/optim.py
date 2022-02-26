class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = [param for param in parameters if param.requires_grad]

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    def step(self):
        raise NotImplementedError(
                'Optimizer class has not implemented the gradient update step yet')


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

