import numpy as np
from typing import List
from functools import reduce
from tqdm import trange

from nn.layers import Layer


class Flatten(Layer):
    def __init__(self):
        super().__init__('Flatten')

    def forward(self, x: np.array):
        super().forward(x)
        self.input = x
        b, _, _, _ = x.shape
        return x.reshape(b, -1)

    def backward(self, d_output: np.array):
        super().backward(d_output)
        return d_output.reshape(self.input.shape)


class Sequential(Layer):
    def __init__(self, layers: List[Layer]):
        super().__init__('Sequential')
        self.layers = layers

    def forward(self, x):
        return reduce(lambda prev_output, layer: layer.forward(prev_output), self.layers, x)

    def backward(self, d_output):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)
        return d_output

    def params(self):
        return reduce(lambda acc, x: acc + x.params(), self.layers, [])


def train(*, net, optimizer, loss, loader, epochs):
    losses = []
    bar = trange(epochs)
    for _ in bar:
        for xs, ys in loader:
            optimizer.zero_grad()

            xs, ys = xs.numpy(), ys.numpy().reshape(xs.shape[0], 1)
            preds = net.forward(xs)
            error = loss.forward(preds, ys)
            net.backward(loss.backward())

            optimizer.step()

            bar.set_description(str(error))
            losses.append(error)
    return losses


def predict(net, loader):
    all_labels = []
    all_preds = []
    for xs, ys in loader:
        xs, ys = xs.numpy(), ys.numpy().reshape(xs.shape[0], 1)

        preds = net.forward(xs)

        all_labels.append(ys.flatten())
        all_preds.append(preds.argmax(axis=1))

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return all_labels, all_preds
