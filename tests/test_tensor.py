import numpy as np
import unittest

from autograd import Tensor


class TestTensor(unittest.TestCase):
    def test_shape(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[1, 2, 3]])
        t3 = Tensor([[1],[2],[3]])
        
        assert t1.shape == (3, )
        assert t2.shape == (1, 3)
        assert t3.shape == (3, 1)
    
    def test_dtype(self):
        t1 = Tensor.ones(8, 1, 4, 4)
        t2 = Tensor([1, 2, 3])
        t3 = Tensor(np.ones((4, 1)), dtype=np.uint8)
        t4 = Tensor.eye(3)
        assert t1.dtype == np.float32
        assert t2.dtype == np.float32
        assert t3.dtype == np.uint8
        assert t4.dtype == np.float32
        t4.uint8()
        assert t4.dtype == np.uint8
        t4 = t4.float32()
        assert t4.dtype == np.float32
    
    def test_inits(self):
        t1 = Tensor.ones(2,4, dtype=np.uint8)
        t2 = Tensor.ones(4,3, dtype=np.float32)
        t3 = Tensor.uniform(3,2,3, dtype=np.uint8)
        t4 = Tensor.uniform(3,2,3, dtype=np.float32)
        
        assert t1.dtype == np.uint8
        assert t2.dtype == np.float32
        assert t3.dtype == np.uint8
        assert t4.dtype == np.float32
        
