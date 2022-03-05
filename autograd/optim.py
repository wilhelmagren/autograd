class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters

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

