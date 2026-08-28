import torch
import torch.nn.functional as F

from src.layers import *


def calc_accuracy(model, pixels, labels):
    confusion_matrix = torch.zeros((10, 10))

    layers = model.layers
    for layer in layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False

    with torch.no_grad():
        x = model(pixels)
        preds = torch.argmax(x, dim=1)
        accuracy = (preds == labels).float().mean()
        for i in range(len(labels)):
            confusion_matrix[labels[i], preds[i]] += 1

    for layer in layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = True

    return accuracy, confusion_matrix


def print_description(epochs, batch_size, optim, alpha, beta,
                       L1_regularization, L2_regularization, corpus_size):
    print(f"epochs: {epochs} |f batch_size: {batch_size} | corpus size: {corpus_size}")
    if L1_regularization:
        print("Regularizer: Lasso")
        print(f"Beta      : {beta}")
    if L2_regularization:
        print("Regularizer:  Ridge")
        print(f"Alpha     : {alpha}")
    optim.describe()


def train_model(iterations, batch_size, model, pixels_train, labels_train, optim, 
                alpha = 0.01, beta = 0.01,
                L1_regularization=False, L2_regularization=False, seed=None):
    # ------------------------------INITIALIZATION AND PRINTING------------------------------
    model.reset_params(seed)
    print_description(iterations, batch_size, optim=optim, alpha=alpha, beta=beta,
                       L1_regularization=L1_regularization, L2_regularization=L2_regularization,
                       corpus_size=labels_train.shape[0])
    
    # ------------------------------- training and backprop -------------------------------
    for step in range(iterations):
        # Forward pass
        ix = torch.randint(0, pixels_train.shape[0], (batch_size,))
        Xb, Yb = pixels_train[ix], labels_train[ix]    # Sample random examples

        optim.zero_grad(set_to_none = True)

        x = model(Xb)   # call model on examples

        # calculate regularizers
        ridge = 0
        lasso = 0
        if L2_regularization:
            for l in model.layers:
                if isinstance(l, Linear):
                    ridge = ridge + (l.weights ** 2).mean()

        if L1_regularization:
            for l in model.layers:
                if isinstance(l, Linear):
                    lasso = lasso + (abs(l.weights)).mean()

        loss = F.cross_entropy(x, Yb) + alpha * ridge + beta * lasso # loss calculation

        # Backward pass
        loss.backward()
        optim.step()

    return model

def build_config(args):
    optim_defaults = {
        "adam" : {
            "lr" : 0.1, 
            "beta1" : 0.999,
            "beta2" : 0.999,
            "eps" : 1e-8
        },
        "rmsprop" : {
            "lr" : 0.01,
            "gamma" : 0.9,
            "eps" : 1e-5
        },
        "adagrad" : {
            "lr" : 0.01, 
            "eps" : 1e-5
        },
        "sgd" : {
            "lr" : 0.001, 
            "momentum" : 0,
            "dampening" : 0
        }
    }

    defaults = optim_defaults[args.optimizer]
    for key, default_val in defaults.items():
        user_val = getattr(args, key, None)
        optim_defaults[key] = user_val if user_val is not None else default_val

    return optim_defaults
