![Workflow status badge](https://github.com/willeagren/autograd/actions/workflows/python-app.yml/badge.svg)

# AUTOGRAD, it is what it is
So I heard you like gradients huh? Well, great! This repository is a small-scale minimalistic deep learning framework inspired by ![PyTorch](https://pytorch.org/), but created to be even more simplistic. It works out-of-the-box and the only real dependency is ![NumPy](https://numpy.org/), except for some minor libraries to fetch datasets. Please see requirements in the setup guide below.

It was created as a means to verify and test my personal knowledge on how practical deep learning frameworks are built and structured. Therefore, I have also created a number of interactive python notebooks (.ipynb) that details the fundamentals of computational graphs for deep learning. I hope that these are informative. 

### Example
```python
from autograd import Tensor

x = Tensor([[-1.4, 2.5, 7.3]], requires_grad=False)
w = Tensor.eye(3)

y = x.dot(w).mean()
y.backward()

print(x.grad)  # dy/dx
print(w.grad)  # dy/dw
```

### Installation
You do not want to even bother installing this. Just use PyTorch instead...

### Running tests
```python
python3 -m unittest
```
### TODO
- more ops [tanh, log, exp, sigmoid, Conv2d, ...]
- implement Adam optimizer
- losses module
