import unittest
import os
import numpy as np
from autograd import Tensor
from autograd.nn.initializers import Constant


class TestInitializers(unittest.TestCase):
    def test_constant(self):
        initializer = Constant(10)
        constant = initializer((2, 3))

        assert constant.shape == (2, 3)

