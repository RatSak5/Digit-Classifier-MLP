import pandas as pd
import torch

# Hyper-params
NUM_CLASSES = 10
NUM_PIXELS = 784


def load_data(train_csv="data/mnist_train_small.csv", test_csv="data/mnist_test.csv", validation=5000):
    df_train = pd.DataFrame(pd.read_csv(train_csv))
    df_test = pd.DataFrame(pd.read_csv(test_csv))

    arr_train = df_train.to_numpy()
    arr_test = df_test.to_numpy()

    pixels_train = arr_train[:, 1:] / 255
    pixels_test = arr_test[:, 1:] / 255

    labels_train = torch.tensor(arr_train[:, 0])
    pixels_train = torch.tensor(pixels_train).float()

    labels_val = torch.tensor(arr_test[:validation, 0])
    pixels_val = torch.tensor(pixels_test[:validation]).float()

    labels_test = torch.tensor(arr_test[validation:, 0])
    pixels_test = torch.tensor(pixels_test[validation:]).float()

    return {
        "pixels_train": pixels_train, "labels_train": labels_train,
        "pixels_val": pixels_val, "labels_val": labels_val,
        "pixels_test": pixels_test, "labels_test": labels_test,
    }


def print_example_counts(labels_train, labels_val, labels_test):
    for name, labels in [("training", labels_train), ("testing", labels_val), ("validating", labels_test)]:
        for i in range(10):
            print(f"Number of {name:10} examples for {i}: ", labels[labels == i].shape[0])
        print("-----------------------------")
