import unittest
import autograd.nn as nn
from autograd import Tensor

class TestNN(unittest.TestCase):
    def test_sequential(self):
        class TestNet(nn.Module):
            def __init__(self):
                self.params = nn.Sequential(
                        nn.Dense(784, 128),
                        nn.Dense(128, 10)
                        )

            def forward(self, x):
                x = self.params(x)
                return x

        model = TestNet()
        params = model.parameters()

        assert len(params) == 2

        for param in params:
            assert isinstance(param, Tensor)
        
        in_x = Tensor.ones(64, 784)
        out_x = model(in_x)

        assert out_x.shape == (64, 10)

    

