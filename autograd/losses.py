import numpy as np

class NLLLoss(object):
    def __call__(self, preds, targets):
        assert preds.shape[0] == targets.shape[0]

        return -preds.data[range(targets.shape[0]), targets.data.astype(np.uint8)].mean()

