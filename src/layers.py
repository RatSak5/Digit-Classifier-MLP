"""
Minimal, from-scratch neural network building blocks implemented on top of
raw PyTorch tensors (no torch.nn). Mirrors the API style of torch.nn modules:
each layer is callable, exposes .parameters(), and (where it holds trainable
state) .reset_params().
"""

import torch


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.bias = bias

        self.weights = torch.randn(fan_in, fan_out) / fan_in ** 0.5  # kaiming
        self.biases = torch.zeros(fan_out) if bias else None

    def __call__(self, inp):
        prod = inp @ self.weights
        if self.biases is not None:
            prod += self.biases
        self.out = prod
        return self.out

    def parameters(self):
        biases = [self.biases] if self.biases is not None else []
        return [self.weights] + biases

    def reset_params(self):
        self.weights = torch.randn(self.fan_in, self.fan_out) / self.fan_in ** 0.5
        self.biases = torch.zeros(self.fan_out) if self.bias else None


class BatchNorm1d:
    def __init__(self, dim, eps=1e-05, momentum=0.1, training=True):
        self.eps = eps
        self.momentum = momentum
        self.dim = dim

        self.training = training

        self.gamma = torch.ones(1, dim)
        self.beta = torch.zeros(1, dim)

        self.running_mean = torch.zeros(1, dim)
        self.running_var = torch.ones(1, dim)

    def __call__(self, inp):
        if self.training:
            bn_mean = inp.mean(dim=0, keepdim=True)
            bn_var = inp.var(dim=0, keepdim=True, unbiased=False)
        else:
            bn_mean = self.running_mean
            bn_var = self.running_var

        inp_norm = (inp - bn_mean) / torch.sqrt(bn_var + self.eps)
        inp_bn = self.gamma * inp_norm + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * bn_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * bn_var

        self.out = inp_bn
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

    def reset_params(self):
        self.gamma = torch.ones(1, self.dim)
        self.beta = torch.zeros(1, self.dim)

        self.running_mean = torch.zeros(1, self.dim)
        self.running_var = torch.ones(1, self.dim)


class Dropout:
    def __init__(self, p=0.5):
        self.distr = torch.tensor([1 - p, p])
        self.p = p

    def __call__(self, inp):
        num_elements = inp.view(-1).shape[0]
        mask = torch.multinomial(self.distr, num_elements, replacement=True).view(inp.shape)
        self.out = mask * inp
        return self.out

    def parameters(self):
        return []


class Tanh:
    def __call__(self, inp):
        self.out = torch.tanh(inp)
        return self.out

    def parameters(self):
        return []


class ReLU:
    def __call__(self, inp):
        self.out = torch.relu_(inp)
        return self.out

    def parameters(self):
        return []


class PReLU:
    def __init__(self):
        self.weight = torch.randn(1)

    def __call__(self, inp):
        self.out = torch.prelu(inp, self.weight)
        return self.out

    def parameters(self):
        return [self.weight]

    def reset_params(self):
        self.weight = torch.randn(1)


class SoftMax:
    def __call__(self, arr):
        logits = torch.exp(arr)
        self.out = logits / logits.sum(dim=1)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, vocab_size, emb_size):
        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.weight = torch.randn((vocab_size, emb_size))

    def __call__(self, inp):
        self.out = self.weight[inp]
        return self.out

    def parameters(self):
        return [self.weight]

    def reset_params(self):
        self.weight = torch.randn((self.vocab_size, self.emb_size))


class Flatten:
    def __call__(self, arr):
        shape = 1 if len(arr.shape) == 2 else arr.shape[0]
        self.out = arr.view(shape, -1)
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.params = [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, inp):
        self.out = inp
        for layer in self.layers:
            self.out = layer(self.out)
        return self.out

    def n_parameters(self):
        return sum(p.nelement() for p in self.params)

    def reset_params(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for layer in self.layers:
            if hasattr(layer, 'reset_params'):
                layer.reset_params()
        self.params = [p for layer in self.layers for p in layer.parameters()]
