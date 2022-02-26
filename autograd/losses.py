import numpy as np
from .tensor import Tensor

class NLLLoss(object):
    def __call__(self, preds, targets):
        assert preds.shape[0] == targets.shape[0]
        return preds.mul(targets).mean()


class CategoricalCrossEntropy(object):
    def __call__(self, logits, targets):
        n_classes = logits.shape[-1]
        if isinstance(targets, np.ndarray):
            tt = targets.flatten()
            y = np.zeros((tt.shape[0], n_classes)).astype(np.float32)
            y[range(y.shape[0]), tt] = -1.0 * n_classes
            y = y.reshape(list(targets.shape) + [n_classes])
            targets = Tensor(y)

        return logits.mul(targets).mean()



