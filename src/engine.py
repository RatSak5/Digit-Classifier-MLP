"""
Training loop with hand-written optimizer update rules (plain SGD, momentum,
AdaGrad, RMSprop, Adam) plus accuracy/confusion-matrix evaluation.
"""

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


def print_description(epochs, batch_size, learn, optim, momentum, beta1, beta2, gamma, eps, alpha, beta,
                       L1_regularization, L2_regularization, corpus_size):
    print(f"epochs: {epochs}, batch_size: {batch_size}, alpha: {alpha}, beta: {beta}, corpus size: {corpus_size}")
    st = ""
    if L1_regularization:
        st += " L1 Regularization valid"
    if L2_regularization:
        st += " L2 Regularization valid"
    optim_dict = {
        "momentum based": f" Momentum based , momentum = {momentum} ",
        "adagrad": f" AdaGrad , eps = {eps} ",
        "rmsprop": f" RMSprop , gamma = {gamma}, eps = {eps} ",
        "adam": f" Adam , Beta1 = {beta1} , beta2 = {beta2} ",
        None: f" Batch GD , learning rate = {learn} ",
    }
    st += optim_dict[optim]
    st += f", learn rate = {learn}"
    print(st)


def train_model(iterations, batch_size, model, pixels_train, labels_train, learn=0.1, optim=None,
                 momentum=0.1, beta1=0.9, beta2=0.9, gamma=0.9, eps=1e-5, alpha=0.01, beta=0.01,
                 L1_regularization=False, L2_regularization=False, seed=None):
    # ------------------------------INITIALIZATION AND PRINTING------------------------------
    model.reset_params(seed)
    print_description(iterations, batch_size, optim=optim,
                       momentum=momentum, beta1=beta1, beta2=beta2, gamma=gamma,
                       eps=eps, alpha=alpha, beta=beta, learn=learn,
                       L1_regularization=L1_regularization, L2_regularization=L2_regularization,
                       corpus_size=labels_train.shape[0])

    for p in model.params:
        p.requires_grad = True

    if optim == "momentum based":
        velocities = [torch.zeros_like(p) for p in model.params]
    if optim == "adagrad":
        accum_grad = [torch.zeros_like(p) for p in model.params]
    if optim == "rmsprop":
        accum_grad_exp = [torch.zeros_like(p) for p in model.params]
    if optim == "adam":
        accum_mt = [torch.zeros_like(p) for p in model.params]
        accum_vt = [torch.zeros_like(p) for p in model.params]

    # ------------------------------- training and backprop -------------------------------
    for step in range(iterations):
        # Forward pass
        ix = torch.randint(0, pixels_train.shape[0], (batch_size,))
        Xb, Yb = pixels_train[ix], labels_train[ix]

        x = model(Xb)

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

        loss = F.cross_entropy(x, Yb) + alpha * ridge + beta * lasso

        # Backward pass
        for p in model.params:
            p.grad = None

        loss.backward()

        # Update step
        lr = learn
        for i in range(len(model.params)):
            if optim == "adam":
                accum_mt[i] = beta1 * accum_mt[i] + (1 - beta1) * model.params[i].grad
                accum_vt[i] = beta2 * accum_vt[i] + (1 - beta2) * model.params[i].grad ** 2

                t = step + 1
                mt_hat = accum_mt[i] / (1 - beta1 ** t)
                vt_hat = accum_vt[i] / (1 - beta2 ** t)

                update = (mt_hat / (vt_hat ** 0.5 + eps)) * lr

            elif optim == "rmsprop":
                accum_grad_exp[i] = gamma * accum_grad_exp[i] + (1 - gamma) * model.params[i].grad ** 2
                update = (model.params[i].grad / (accum_grad_exp[i] + eps) ** 0.5) * lr

            elif optim == "momentum based":
                velocities[i] = lr * model.params[i].grad + momentum * velocities[i]
                update = velocities[i]

            elif optim == "adagrad":
                accum_grad[i] = accum_grad[i] + model.params[i].grad ** 2
                update = (model.params[i].grad / (accum_grad[i] + eps) ** 0.5) * lr

            else:
                update = lr * model.params[i].grad

            model.params[i].data += -update

    return model
