from autograd import Tensor


class NN(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        tensors = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                if attr.requires_grad:
                    tensors.append(attr)
        return tensors
    
    def forward(self, x):
        raise NotImplementedError(
        f'user defined NN object has not implemented forward pass')

