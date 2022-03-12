import os
import numpy as np
import unittest
from tqdm import trange
from autograd import Tensor, SGD, NLLLoss, fetch_mnist
from autograd import Adam
import autograd.nn as nn
np.random.seed(1)


__optimizers__ = {}
__optimizers__['sgd'] = SGD
__optimizers__['adam'] = Adam


class MNISTNet(nn.Module):
    def __init__(self):
        self.affines = nn.Sequential(
                nn.Dense(784, 128),
                nn.ReLU(),
                nn.Dense(128, 10),
                nn.LogSoftmax()
                )

    def forward(self, x):
        return self.affines(x)

   
class TestMNISTdigits(unittest.TestCase):
    def test(self):
        def _test(optimiz):
            model = MNISTNet()
            criterion = NLLLoss()
            optimizer = __optimizers__[optimiz](model.parameters(), lr=1e-3)
            X_train, Y_train, X_test, Y_test = fetch_mnist()
            X_train = X_train.reshape((-1, 784))
            X_test = X_test.reshape((-1, 784))
            epochs = 300
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
            print(f'optimizer {optimiz} got {100*acc:.1f} % acc')

        _test('sgd')
        _test('adam')


class TestMNISTfashion(unittest.TestCase):
    def test(self):
        def _test(optimiz):
            model = MNISTNet()
            criterion = NLLLoss()
            optimizer = __optimizers__[optimiz](model.parameters(), lr=1e-3)
            X_train, Y_train, X_test, Y_test = fetch_mnist('fashion')
            X_train = X_train.reshape((-1, 784))
            X_test = X_test.reshape((-1, 784))
            epochs = 300
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
            print(f'optimizer {optimiz} got {100*acc:.1f} % acc')

        _test('sgd')
        _test('adam')

