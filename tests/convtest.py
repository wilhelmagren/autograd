import numpy as np
import autograd.nn as nn
from autograd import Tensor, fetch_mnist
from autograd import Adam, NLLLoss

class MNISTCNN(nn.Module):
    def __init__(self):
        self.convs_ = nn.Sequential(
                nn.Conv2d(8, 1, 3, 3),
                nn.ReLU(),
                nn.Conv2d(16, 8, 3, 3),
                nn.ReLU()
                )
                
        self.affines_ = nn.Sequential(
                nn.Dense(1, 64),
                nn.ReLU(),
                nn.Dense(64, 10),
                nn.LogSoftmax()
                )

    def forward(self, x):
        x = self.convs_(x)
        x = x.reshape(shape=(x.shape[0], -1))
        x = self.affines_(x)

        return x


X_train, Y_train, X_test, Y_test = fetch_mnist()
X_train = X_train.reshape((-1, 1, 28, 28))
X_test = X_test.reshape((-1, 1, 28, 28))
model = MNISTCNN()
optimizer = Adam(model.parameters())
criterion = NLLLoss()
n_epochs = 300
batch_size = 128

for _ in range(n_epochs):
    indices = np.random.randint(0, X_train.shape[0], size=(batch_size))
    samples = Tensor(X_train[indices])
    targets = Y_train[indices]

    logits = model(samples)
    loss = criterion(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = np.argmax(logits.data, axis=-1)
    acc = (preds == targets).mean()

    print(f'loss {loss.data[0]:.2f}  acc {acc:.2f}')
