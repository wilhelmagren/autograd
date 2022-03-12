import os
import torch
import numpy as np
import timeit
import unittest
from autograd import Tensor
from functools import partial


def _test_op(shapes, torch_func, autograd_func, name):
    torch_tensors = [torch.tensor(np.random.random(size=shape), requires_grad=True) for shape in shapes]
    autograd_tensors = [Tensor(tensor.detach().numpy()) for tensor in torch_tensors]

    torch_out = torch_func(*torch_tensors)
    autograd_out = autograd_func(*autograd_tensors)
    
    np.testing.assert_allclose(torch_out.detach().numpy(),
            autograd_out.data, atol=1e-6, rtol=1e-3)

    torch_forward = timeit.Timer(partial(torch_func, *torch_tensors))
    f_torch_ms = torch_forward.timeit(10) * 1000.0 / 10.0
    autograd_forward = timeit.Timer(partial(autograd_func, *autograd_tensors))
    f_autograd_ms = autograd_forward.timeit(10) * 1000.0 / 10.0

    torch_backward = timeit.Timer(partial(lambda f, x: f(*x).mean().backward(), torch_func, torch_tensors))
    b_torch_ms = torch_backward.timeit(10) * 1000.0 / 10.0
    autograd_backward = timeit.Timer(partial(lambda f, x: f(*x).mean().backward(), autograd_func, autograd_tensors))
    b_autograd_ms = autograd_backward.timeit(10) * 1000.0 / 10.0
    
    for tt, at in zip(torch_tensors, autograd_tensors):
        np.testing.assert_allclose(tt.grad / 10.0, at.grad, atol=1e-6, rtol=1e-3)

    print(f'\ntesting {name} with shapes {shapes}, torch/autograd \n'\
          f'forward: {f_torch_ms:.2f} ms / {f_autograd_ms:.2f} ms   '\
          f'backward: {b_torch_ms:.2f} ms / {b_autograd_ms:.2f} ms\n')


class TestOps(unittest.TestCase):
    def test_mean(self):
        _test_op([(56, 82)], lambda x: x.mean(), Tensor.mean, 'mean')

    def test_sum(self):
        _test_op([(56, 82)], lambda x: x.sum(), Tensor.sum, 'sum')

    def test_add(self):
        _test_op([(56, 82), (56, 82)], lambda x, y: x + y, Tensor.add, 'add')

    def test_sub(self):
        _test_op([(56, 82), (56, 82)], lambda x, y: x - y, Tensor.sub, 'sub')

    def test_mul(self):
        _test_op([(56, 82), (56, 82)], lambda x, y: x * y, Tensor.mul, 'mul')

    def test_dot(self):
        _test_op([(56, 82), (82, 45)], lambda x, y: x.matmul(y), Tensor.dot, 'dot')

    def test_relu(self):
        _test_op([(56, 82)], lambda x: x.relu(), Tensor.relu, 'relu')

    def test_logsoftmax(self):
        _test_op([(56, 82)], lambda x: torch.nn.LogSoftmax(dim=1)(x), Tensor.logsoftmax, 'logsoftmax')

    def test_exp(self):
        _test_op([(56, 82)], lambda x: x.exp(), Tensor.exp, 'exp')

    def test_log(self):
        _test_op([(56, 82)], lambda x: x.log(), Tensor.log, 'log')

    def test_sigmoid(self):
        _test_op([(56, 82)], lambda x: x.sigmoid(), Tensor.sigmoid, 'sigmoid')

