![Unit Tests Workflow Status Badge](https://github.com/willeagren/autograd/actions/workflows/unittests.yml/badge.svg) ![Count Lines of Code Workflow Status Badge](https://github.com/willeagren/autograd/blob/main/images/cloc.svg) [![GitLicense](https://gitlicense.com/badge/willeagren/autograd)](https://gitlicense.com/license/willeagren/autograd)

# AUTOGRAD, it is what it is
So I heard you like gradients huh? Well, great! This repository is a small-scale minimalistic deep learning framework inspired by [PyTorch](https://pytorch.org/), but created to be even more simplistic. It works out-of-the-box and the only real dependency is [NumPy](https://numpy.org/), except for some minor libraries to fetch datasets. Please see requirements in the setup guide below.

It was created as a means to verify and test my personal knowledge on how practical deep learning frameworks are built and structured. Therefore, I have also created a number of interactive python notebooks (.ipynb) that details the fundamentals of computational graphs for deep learning. I hope that these are informative. 

### Example
```python
from autograd import Tensor

x = Tensor([[-1.4, 2.5, 7.3]])
w = Tensor.eye(3)

y = x.dot(w).mean()
y.backward()

print(x.grad)  # dy/dx
print(w.grad)  # dy/dw
```

### Neural networks
It's really simple actually, just import the `autograd.nn` modules and start working! Specify model parameters as attributes of the neural network class and implement a forward pass for your model. You yield outputs by calling your initialized neural network class. Below is an example pipeline for MNIST digits classification.
```python
import autograd.nn as nn
import numpy as np
from autograd import Tensor, fetch_mnist
from autograd import Adam, NLLLoss

# define neural network architecture
class MNISTClassifier(nn.Module):
  def __init__(self):
    self.affines_ = nn.Sequential(
        nn.Dense(784, 128),
        nn.ReLU(),
        nn.Dense(128, 10),
        nn.LogSoftmax()
        )
  
  def forward(self, x):
    return self.affines_(x)

# load MNIST digits data
X_train, Y_train, X_test, Y_test = fetch_mnist()
model = MNISTClassifier()
optimizer = SGD(model.parameters())
criterion = NLLLoss()

for _ in range(epochs):
  logits = model(Tensor(X_train))
  loss = criterion(logits, Tensor(Y_train))
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# calculate final model score
Y_test_preds = np.argmax(model(Tensor(X_test)).data, axis=-1)
acc = (Y_test_preds == Y_test).mean()
```

### Installation
Just install PyTorch instead.

### Running tests
```python
python3 -m unittest
```
### TODO
- more ops [tanh, Conv2d, ...]
- losses module
- write installation steps and requirements
