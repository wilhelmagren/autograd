![Workflow status badge](https://github.com/willeagren/autograd/actions/workflows/python-app.yml/badge.svg)
# AUTOGRAD, it is what it is
Do you like gradients? Do you like automation? Do you like deep learning? Well, great, because this is a small library for automatic computation of analytical gradients. As per all deep learning frameworks, a directed acyclic graph (DAG) is constructed when applying operations to your tensors which can be utilized to perform backpropagation. If you do not know how any of this works, see the interactive python notebook ![computational_graphs](notebooks/computational_graphs.ipynb) that goes through some theory and a practical example. ❤️

I think it was Master Oogway that once said "If you ever want to verify your theoretical knowledge, do it practically.", or something like that. So I decided to try and impplement what I believed to master; and I dare say that the results are at least somewhat good... (except for performance, memory management, and all that stuffs)

### Example
```python
from autograd.tensor import Tensor

x = Tensor([[-1.4, 2.5, 7.3]], requires_grad=False)
w = Tensor.eye(3)

y = x.matmul(w).mean()
y.backward()

# Since y is a tensor yielded as a results of ops,
# it requires to be tied to a context of a function.
assert y._ctx is not None

print(x.grad)  # dy/dx
print(w.grad)  # dy/dw
```

### Installation
You do not want to even bother installing this. Just use PyTorch instead...

### TODO
- more ops [ReLU, tanh, dot, log, ...]
- implement optimizers [Adam, SGD, adagrad, ...]
- neural networks
- losses
