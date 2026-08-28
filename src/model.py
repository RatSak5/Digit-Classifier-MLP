import torch
from src.data import NUM_CLASSES, NUM_PIXELS
from src.layers import *

gain = {"Tanh": (5 / 3) ** 0.5, "ReLU": 2 ** 0.5}


def build_model(n_hidden=260, n_pixels=NUM_PIXELS, n_classes=NUM_CLASSES):
    model = Sequential([
        Linear(n_pixels, n_hidden, bias=False), BatchNorm1d(n_hidden), PReLU(),
        Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), PReLU(),
        Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), PReLU(),
        Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), PReLU(),
        Linear(n_hidden, n_classes, bias=False), BatchNorm1d(n_classes),
    ])

    n = len(model.layers)
    with torch.no_grad():
        if isinstance(model.layers[-1], Linear):
            model.layers[-1].weights *= 0.01
        elif isinstance(model.layers[-1], BatchNorm1d):
            model.layers[-1].gamma *= 0.01

        for i in range(n - 1):
            layer = model.layers[i]
            if isinstance(layer, Linear) and i < n - 1 and model.layers[i + 1].__class__.__name__ in gain:
                    layer.weights *= gain[model.layers[i + 1].__class__.__name__]
            if isinstance(layer, BatchNorm1d):
                layer.training = True

            model.layers[i] = layer
    
    for p in model.params:
        p.requires_grad = True

    return model
