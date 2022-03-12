"""----------------------------------------------------------
Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 12-03-2022
License: MIT
----------------------------------------------------------"""

from autograd import Tensor


class Module(object):
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError(
        f'user defined nn.Module object has not implemented forward pass')

    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                if attr.requires_grad:
                    params.append(attr)
            elif isinstance(attr, list):
                params.extend([param for param in attr if param.requires_grad])
            elif isinstance(attr, Sequential):
                params.extend([param for param in attr.parameters() if param.requires_grad])
        return params


class Sequential(object):
    def __init__(self, *modules):
        self.modules_ = modules
    
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        for module in self.modules_:
            x = module(x)
        return x
    
    def parameters(self):
        params = []
        for module in self.modules_:
            params.extend(module.parameters())
        return params


