![Workflow status badge](https://github.com/willeagren/autograd/actions/workflows/python-app.yml/badge.svg)

# AUTOGRAD, it is what it is
So I heard you like gradients huh? Well, great! This repository is a small-scale minimalistic deep learning framework inspired by [PyTorch](https://pytorch.org/), but created to be even more simplistic. It works out-of-the-box and the only real dependency is [NumPy](https://numpy.org/), except for some minor libraries to fetch datasets. Please see requirements in the setup guide below.

It was created as a means to verify and test my personal knowledge on how practical deep learning frameworks are built and structured. Therefore, I have also created a number of interactive python notebooks (.ipynb) that details the fundamentals of computational graphs for deep learning. I hope that these are informative. 

### Example
```python
from autograd import Tensor

# since x is our input data we do not want to change its 
# values, hence, no gradient required for it.
x = Tensor([[-1.4, 2.5, 7.3]], requires_grad=False)
w = Tensor.eye(3)

y = x.dot(w).mean()
y.backward()

print(x.grad)  # dy/dx
print(w.grad)  # dy/dw
```

### Neural networks
It's really simple, just import the NN class and inherit it with your user defined network. Specify your model parameters as attributes of the class and implement a forward pass. Yield model outputs by invoking a call to the model with your input data. An example pipeline can be seen below.
```python
from autograd import Tensor, fetch_mnist
from autograd import SGD, NLLLoss, NN

class Net(NN):
  def __init__(self):
    self.l1 = Tensor.uniform(784, 128)
    self.l2 = Tensor.uniform(128, 10)
  
  def forward(self, x):
    # implement forward pass however you want
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

# fetch data
X_train, Y_train, X_test, Y_test = fetch_mnist()
model = Net()
optimizer = SGD(model.parameters())
criterion = NLLLoss()

for _ in epochs:
  logits = model(X_train)
  loss = criterion(logits, Y_train)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# and calculate final model score below, like in PyTorch
```

### Installation
Just install PyTorch instead.

### Running tests
```python
python3 -m unittest
```
### TODO
- more ops [tanh, log, exp, sigmoid, Conv2d, ...]
- implement Adam optimizer
- losses module
- write installation steps and requirements
