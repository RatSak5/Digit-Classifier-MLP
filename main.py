import argparse
import copy

import torch
from src.data import load_data, print_example_counts
from src.engine import build_config, calc_accuracy, train_model
from src.model import build_model
from src.optim import *
from src.visualize import plot_confusion_matrices

seed = 232325325  #for reproducability


#creating parser object for passing inputs through command line
def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP digit classifier on MNIST.")
    parser.add_argument("--train-csv", default="data/mnist_train_small.csv")
    parser.add_argument("--test-csv", default="data/mnist_test.csv")
    parser.add_argument("--iterations", type=int, default=14000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-hidden", type=int, default=260)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop", "adagrad"],
                        type=str, default="adam")

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()
    #For better visibility (basically 0.1 instead of 1e-1)
    torch.set_printoptions(sci_mode=False)
    #Create the data
    data = load_data(args.train_csv, args.test_csv, validation=args.val_size)
    print_example_counts(data["labels_train"], data["labels_val"], data["labels_test"])
    #Change stuff if you like.
    model = build_model(n_hidden=args.n_hidden)
    print(f"Num Params : {model.n_parameters()}")
    matrices, titles, models = [], [], {}
    
    optimizer = args.optimizer
    config = build_config(args)
    optim = Adam(model.params, **config[optimizer])

    # Example states
    input_state = {
        "iterations"       : args.iterations, 
        "batch_size"       : args.batch_size, 
        "model"            : model, 
        "pixels_train"     : data["pixels_train"], 
        "labels_train"     : data["labels_train"], 
        "optim"            : optim, 
        "L2_regularization": True, 
        "seed"             : seed
    }
    trained_model = train_model(**input_state)
    #Accuracies
    train_acc, _ = calc_accuracy(trained_model, data["pixels_train"], data["labels_train"])
    val_acc, conf_mx_val = calc_accuracy(trained_model, data["pixels_val"], data["labels_val"])
    #print accuracies
    print(f"Accuracy of model {optim.label} on train : {train_acc * 100}%")
    print(f"Accuracy of model {optim.label} on  eval : {val_acc * 100}%")
    #save stuff related to model
    matrices.append(conf_mx_val)
    titles.append(f"Model ({optim.label})")
    models[optim.label] = copy.deepcopy(trained_model)

    #Test set evaluation on any models you like.
    model_ = models[optim.label]
    accuracy, conf_mx_test = calc_accuracy(model_, data["pixels_test"], data["labels_test"])
    print(f"Accuracy on {optim.label} model: {accuracy.item() * 100}%")
    print("Confusion matrix :")
    print(conf_mx_test)

    #Plot and compare!
    plot_confusion_matrices(matrices, titles)


if __name__ == "__main__":
    main()