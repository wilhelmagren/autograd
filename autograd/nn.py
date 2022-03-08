from autograd import Tensor


class NN(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        pass
    
    def forward(self, x):
        raise NotImplementedError(
        f'user defined NN object has not implemented forward pass')

