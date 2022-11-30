import numpy as np


class MSELoss:
    def __init__(self):
        self.grad = None

    def forward(self, preds, ys):
        self.grad = 2 * (preds - ys) / ys.size
        return ((preds - ys) ** 2).mean()

    def backward(self):
        return self.grad


class CELoss:
    def __init__(self):
        self.grad = None

    def forward(self, preds, ys):
        mask = np.zeros_like(preds)
        mask[np.arange(len(ys)), ys.flatten()] = 1

        self.grad = np.where(mask == 1, -1 / preds, 0)
        return (np.where(mask == 1, -np.log(preds), 0)).sum() / preds.shape[0]

    def backward(self):
        return self.grad
