import numpy as np
from autograd import Tensor

class Criterion(object):
    def __call__(self, preds, targets):
        return self.apply(preds, targets)

    def apply(self, *args, **kwargs):
        raise NotImplementedError(
        f'user defined Criterion object has not implemented apply method')


class NLLLoss(Criterion):
    """implements the SparseCategoricalCrossEntroyLoss
    by getting predictions that have been logsoftmaxed 
    prior to calling the loss function. targets should be
    indices of the target class.
    """
    def apply(self, logits, targets):
        n_classes = logits.shape[-1]
        y = np.zeros((targets.shape[0], n_classes)).astype(np.float32)
        y[range(y.shape[0]), targets] = -1.0 * n_classes
        targets = Tensor(y)
        return logits.mul(targets).mean()

