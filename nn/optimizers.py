import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        for _, grad in self.params:
            grad[:] = 0


class SGD(Optimizer):
    def step(self):
        for param, grad in self.params:
            param[:] -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, params, lr, betas=(0.99, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.cache = {}

    def step(self):
        b1, b2 = self.betas

        for i, (param, grad) in enumerate(self.params):
            param_grad = grad

            running_avg_grad, norm_grad = self.cache.get(i, (param_grad, param_grad ** 2))

            running_avg_grad = b1 * running_avg_grad + (1 - b1) * param_grad
            norm_grad = self.__get_grad_norm(norm_grad, param_grad)

            self.cache[i] = (running_avg_grad, norm_grad)

            param[:] -= self.lr * self.__gradient(i)

    def __gradient(self, i):
        running_avg_grad, l2_norm_grad = self.cache[i]
        return running_avg_grad / np.sqrt(l2_norm_grad + self.eps)

    def __get_grad_norm(self, norm_grad, param_grad):
        return self.betas[1] * norm_grad + (1 - self.betas[1]) * param_grad ** 2


class AdaMax(Adam):
    def __gradient(self, i):
        running_avg_grad, norm_grad = self.cache[i]
        return running_avg_grad / norm_grad

    def get_grad_l2(self, norm_grad, param_grad):
        _, b2 = self.betas
        return np.maximum(b2 * norm_grad, np.abs(param_grad) + self.eps)
