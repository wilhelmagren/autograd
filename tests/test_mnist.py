import os
import numpy as np
import unittest
from tqdm import trange
from autograd import Tensor, NN, SGD, NLLLoss, fetch_mnist


class MNISTNet(NN):
    def __init__(self):
        self.l1 = Tensor.uniform(784, 128)
        self.l2 = Tensor.uniform(128, 10)

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


class TestMNISTdigits(unittest.TestCase):
    def test(self):
        model = MNISTNet()
        criterion = NLLLoss()
        optimizer = SGD(model.parameters(), lr=1e-3)
        X_train, Y_train, X_test, Y_test = fetch_mnist()
        X_train = X_train.reshape((-1, 784))
        X_test = X_test.reshape((-1, 784))
        epochs = 200
        batch_size = 128

        for _ in (t := trange(epochs, disable=os.getenv('CI') is not None)):
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

            t.set_description(
            f'loss {loss.data[0]:.2f}  accuracy {acc:.2f}')

        Y_test_preds_out = model(Tensor(X_test)).data
        Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
        acc = (Y_test_preds == Y_test).mean()

        assert acc >= 0.9


class TestMNISTfashion(unittest.TestCase):
    def test(self):
        model = MNISTNet()
        criterion = NLLLoss()
        optimizer = SGD(model.parameters(), lr=1e-3)
        X_train, Y_train, X_test, Y_test = fetch_mnist('fashion')
        X_train = X_train.reshape((-1, 784))
        X_test = X_test.reshape((-1, 784))
        epochs = 200
        batch_size = 128

        for _ in (t := trange(epochs, disable=os.getenv('CI') is not None)):
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

            t.set_description(
            f'loss {loss.data[0]:.2f}  accuracy {acc:.2f}')

        Y_test_preds_out = model(Tensor(X_test)).data
        Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
        acc = (Y_test_preds == Y_test).mean()

        assert acc >= 0.75

