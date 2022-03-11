import unittest
from autograd import Tensor


class TestNN(unittest.TestCase):
    def test_model(self):
        class TestNet(object):
            def __init__(self):
                self.l1 = Tensor.uniform(784, 128)
                self.l2 = Tensor.uniform(128, 10)
                self.l3 = Tensor.ones(100, 100, requires_grad=False)

            def __call__(self, x):
                return self.forward(x)

            def forward(self, x):
                return x.dot(self.l1).relu().dot(self.l2)

            def parameters(self):
                tensors = []
                for attr in self.__dict__.values():
                    if isinstance(attr, Tensor):
                        if attr.requires_grad:
                            tensors.append(attr)
                return tensors

        model = TestNet()
        params = model.parameters()

        assert len(params) == 2

        for param in params:
            assert isinstance(param, Tensor)
        
        in_x = Tensor.ones(64, 784)
        out_x = model(in_x)

        assert out_x.shape == (64, 10)

    

