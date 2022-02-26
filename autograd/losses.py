class NLLLoss(object):
    def __call__(self, preds, targets):
        assert preds.shape[0] == targets.shape[0]
        return preds.mul(targets).mean()

