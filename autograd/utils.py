"""
"""
from functools import partialmethod
from .tensor import Tensor


def register(name, func):
    setattr(Tensor, name, partialmethod(func.apply, func))

