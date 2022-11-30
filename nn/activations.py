import numpy as np
from scipy.special import softmax
from nn.layers import Layer


class Sigmoid(Layer):
    def __init__(self):
        super().__init__('Sigmoid')
        self.sigma = lambda x: 1 / (1 + np.exp(-x))

    def forward(self, x: np.array):
        super().forward(x)
        self.input = x
        return self.sigma(x)

    def backward(self, d_output: np.array):
        super().backward(d_output)
        return d_output * self.sigma(self.input) * (1 - self.sigma(self.input))


class ReLU(Layer):
    def __init__(self, c=0.1):
        super().__init__('ReLU')
        self.c = c

    def forward(self, x: np.array):
        super().forward(x)
        self.input = x
        res = x
        res[res < 0] *= self.c
        return res

    def backward(self, d_output: np.array):
        super().backward(d_output)
        d_output[self.input < 0] *= self.c
        return d_output


class SoftMax(Layer):
    def __init__(self):
        super().__init__('SoftMax')
        self.res = None

    def forward(self, x: np.array):
        self.res = np.exp(x) / np.exp(x).sum(axis=1)[:, None]
        self.res += 1e-8
        return self.res

    def backward(self, dx):
        return self.res * (dx - (dx * self.res).sum(axis=1)[:, None])
