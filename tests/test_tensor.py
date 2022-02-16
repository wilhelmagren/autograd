import numpy as np
import unittest

from autograd.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_shape(self):
        t1 = Tensor([[1.0], [3.2]])
        t2 = Tensor([[2.4, -6.1]])
        t3 = Tensor([-15.2, 123.5])

        assert t1.shape == (2, 1)
        assert t2.shape == (1, 2)
        assert t3.shape == (2, )

    def test_dtype(self):
        t1 = Tensor([1.3, -5.2, 9.6])
        t2 = Tensor(np.array([1, 2, 3], dtype=np.int8))

        assert t1.dtype == np.float32
        assert t2.dtype == np.int8
    
    def test_zeros(self):
        t1 = Tensor.zeros(1, 3)
        t2 = Tensor.zeros(3)
        t3 = Tensor.zeros(4, 1, 2)

        assert t1.shape == (1, 3)
        assert t2.shape == (3, )
        assert t3.shape == (4, 1, 2)
        assert np.all(t1.data == 0.0)
        assert np.all(t2.data == 0.0)
        assert np.all(t3.data == 0.0)
    
    def test_ones(self):
        t1 = Tensor.ones(1, 3)
        t2 = Tensor.ones(3)
        t3 = Tensor.ones(4, 1, 2)
        t4 = Tensor(np.ones((1, 2, 3), dtype=np.int8))

        assert t1.shape == (1, 3)
        assert t2.shape == (3, )
        assert t3.shape == (4, 1, 2)
        assert t4.shape == (1, 2, 3)
        assert np.all(t1.data == 1.0)
        assert np.all(t2.data == 1.0)
        assert np.all(t3.data == 1.0)
        assert np.all(t4.data == 1)

    def test_eye(self):
        t1 = Tensor.eye(1)
        t2 = Tensor.eye(3)

        assert t1.shape == (1, 1)
        assert t2.shape == (3, 3)
        
        
    def test_randn(self):
        t1 = Tensor.randn(1)
        t2 = Tensor.randn(1,3,6,4)

        assert t1.shape == (1, )
        assert t2.shape == (1, 3, 6, 4)
