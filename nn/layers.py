import numpy as np
from scipy.signal import correlate2d


class Layer:
    def __init__(self, name):
        self.name = name
        self.input = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # print(f'Forward {self.name}: ', x.shape)
        pass

    def backward(self, dx):
        # print(f'Backward {self.name}: ', dx.shape)
        pass

    def params(self):
        return []


class Linear(Layer):
    def __init__(self, n_input: int, n_output: int):
        super().__init__(name='Linear')
        self.W = np.random.normal(loc=0.0, scale=np.sqrt(2.0 / n_input), size=(n_input, n_output))
        self.b = np.zeros((1, n_output)) + 0.01
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x: np.array):
        super().forward(x)
        self.input = x
        return x @ self.W + self.b

    def backward(self, d_output: np.array):
        super().backward(d_output)
        d_input = d_output @ self.W.T

        self.grad_W[:] = self.input.T @ d_output
        self.grad_b[:] = d_output.sum(axis=0, keepdims=True)
        return d_input

    def params(self):
        return [(self.W, self.grad_W), (self.b, self.grad_b)]


class Conv2D(Layer):
    def __init__(self, *, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__('Conv2D')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernels = np.random.normal(loc=0.0, scale=np.sqrt(2.0 / in_channels),
                                        size=(out_channels, in_channels, kernel_size, kernel_size))
        self.biases = np.zeros(out_channels) + 0.01
        self.stride = stride
        self.kernel_size = kernel_size

        self.grad_kernels = np.zeros_like(self.kernels)
        self.grad_biases = np.zeros_like(self.biases)

    def forward(self, x: np.array):
        super().forward(x)
        self.input = x

        batches, in_channels, h, w = x.shape
        out_h = int((h - self.kernel_size) / self.stride) + 1
        out_w = int((w - self.kernel_size) / self.stride) + 1

        result = np.zeros((batches, self.out_channels, out_h, out_w))

        for i, to_conv in enumerate(x):
            for j, kernel in enumerate(self.kernels):
                for k, in_channel in enumerate(to_conv):
                    kern = kernel[k]
                    to_add = correlate2d(in_channel, kern, mode="valid")
                    result[i, j] += to_add[::self.stride, ::self.stride]
                result[i, j] += self.biases[j]

        return result

    def backward(self, d_output: np.array):
        super().backward(d_output)

        d_input = np.zeros(self.input.shape)
        self.grad_kernels[:] = 0
        self.grad_biases[:] = 0

        _, _, h_out, w_out = d_output.shape

        for i, inp in enumerate(self.input):
            for j in range(self.out_channels):
                for k, in_channel in enumerate(inp):

                    for ii in range(h_out):
                        for jj in range(w_out):
                            ii_from = ii * self.stride
                            ii_to = ii_from + self.kernel_size
                            jj_from = jj * self.stride
                            jj_to = jj_from + self.kernel_size

                            d_output_scalar = d_output[i, j, ii, jj]
                            d_input[i, k, ii_from:ii_to, jj_from:jj_to] = d_output_scalar * self.kernels[j, k]

                            input_slice = inp[k, ii_from:ii_to, jj_from:jj_to]
                            self.grad_kernels[j, k] += input_slice * d_output_scalar

                            self.grad_biases[j] += d_output_scalar
        return d_input

    def params(self):
        return [(self.kernels, self.grad_kernels), (self.biases, self.grad_biases)]


class Pool2D(Layer):
    def __init__(self, *, kernel_size, stride, operation):
        super().__init__('Pool2D')
        self.kernel_size = kernel_size
        self.stride = stride
        self.operation = operation

    def forward(self, x: np.array):
        super().forward(x)
        batches, channels, h, w = x.shape

        out_h = int((h - self.kernel_size) / self.stride) + 1
        out_w = int((w - self.kernel_size) / self.stride) + 1

        result = np.zeros((batches, channels, out_h, out_w))

        for idx, inp in enumerate(x):
            for ch, channel in enumerate(inp):
                for i in range(out_h):
                    for j in range(out_w):
                        i_from = i * self.stride
                        i_to = i_from + self.kernel_size
                        j_from = j * self.stride
                        j_to = j_from + self.kernel_size

                        patch = channel[i_from:i_to, j_from:j_to]
                        ii, jj, elem = self.operation(patch)

                        result[idx, ch, i, j] = elem
                        yield idx, ch, i_from, j_from, ii, jj
        return result


class MaxPool2D(Pool2D):
    def __init__(self, *, kernel_size, stride):
        def op(patch):
            index = np.argmax(patch)
            ii, jj = np.unravel_index(index, patch.shape)
            max_elem = patch[ii, jj]
            return ii, jj, max_elem

        super().__init__(kernel_size=kernel_size, stride=stride, operation=op)
        self.mask = None

    def forward(self, x: np.array):
        self.mask = np.zeros_like(x, dtype=float)
        gen = super().forward(x)
        try:
            while True:
                idx, ch, i_from, j_from, ii, jj = next(gen)
                self.mask[idx, ch, i_from + ii, j_from + jj] = 1.0
        except StopIteration as ex:
            return ex.value

    def backward(self, d_output):
        super().backward(d_output)
        for i, inp in enumerate(d_output):
            for j, ch in enumerate(inp):
                for k, y in enumerate(ch):
                    for t, x in enumerate(y):
                        i_from = k * self.stride
                        i_to = i_from + self.kernel_size
                        j_from = t * self.stride
                        j_to = j_from + self.kernel_size
                        self.mask[i, j, i_from:i_to, j_from:j_to] *= x
        return self.mask


class AvgPool2D(Pool2D):
    def __init__(self, *, kernel_size, stride):
        def op(patch):
            return -1, -1, patch.mean()

        super().__init__(kernel_size=kernel_size, stride=stride, operation=op)
        self.backward_output = None

    def forward(self, x: np.array):
        self.backward_output = np.zeros_like(x, dtype=float)
        gen = super().forward(x)
        try:
            while True:
                next(gen)
        except StopIteration as ex:
            return ex.value

    def backward(self, d_output):
        super().backward(d_output)
        for i, x in enumerate(d_output):
            for j, channel in enumerate(x):
                for k, y in enumerate(channel):
                    for t, elem in enumerate(y):
                        i_from = k * self.stride
                        i_to = i_from + self.kernel_size
                        j_from = t * self.stride
                        j_to = j_from + self.kernel_size
                        self.backward_output[i, j, i_from:i_to, j_from:j_to] += elem / (self.kernel_size ** 2)
        return self.backward_output


class Add(Layer):
    def __init__(self):
        super().__init__('Add')
        self.left = None
        self.right = None

    def forward(self, x: np.array):
        assert len(x) == 2
        self.left = x[0]
        self.right = x[1]

        return self.left + self.right

    def backward(self, d_output: np.array):
        return np.array([d_output, d_output])


class ResidualBlock(Layer):
    def __init__(self, layers, downsample):
        super().__init__('ResidualBlock')
        self.layers = layers
        self.add_layer = Add()
        self.downsample = downsample

    def forward(self, x: np.array):
        self.input = x

        fwd = self.layers(x)
        return self.add_layer([fwd, self.downsample(self.input)])

    def backward(self, d_output: np.array):
        dl, dr = self.add_layer.backward(d_output)
        d_path = self.layers.backward(dl)
        d_downsample = self.downsample.backward(dr)

        return d_path + d_downsample

    def params(self):
        return self.layers.params() + self.downsample.params()
