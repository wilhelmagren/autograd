import unittest

from autograd import Tensor, NN


class TestNN(unittest.TestCase):
    def test_model(self):
        class TestNet(NN):
            def __init__(self):
                self.l1 = Tensor.uniform(784, 128)
                self.l2 = Tensor.uniform(128, 10)

            def forward(self, x):
                return x.dot(self.l1).relu().dot(self.l2)

        model = TestNet()
        params = model.parameters()

        for param in params:
            assert isinstance(param, Tensor)
        
        in_x = Tensor.ones(64, 784)
        out_x = model(in_x)

        assert out_x.shape == (64, 10)

    

