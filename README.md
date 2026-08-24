# Digit Classification (MNIST) — Neural Net From Scratch

A multi-layer perceptron for classifying MNIST digits, built directly on
raw PyTorch tensors rather than `torch.nn`. Custom implementations include:

- **Layers**: `Linear`, `BatchNorm1d`, `Dropout`, `Tanh`, `ReLU`, `PReLU`,
  `SoftMax`, `Embedding`, `Flatten`, and a `Sequential` container
- **Optimizers**: plain SGD, Momentum, AdaGrad, RMSprop, and Adam — all
  hand-written update rules inside the training loop
- **Regularization**: optional L1 / L2 penalties on linear layer weights

The training script trains the same architecture under all five optimizers
and compares validation/test accuracy and confusion matrices.

## Project structure

```
digit-classification/
├── main.py               # entry point: trains + evaluates all optimizer configs
├── requirements.txt
├── data/                  # put mnist_train_small.csv / mnist_test.csv here
├── notebooks/
│   └── Digit_classification.ipynb   # original exploratory notebook
└── src/
    ├── layers.py          # Linear, BatchNorm1d, activations, Sequential
    ├── model.py            # architecture definition + custom weight init
    ├── data.py             # CSV loading / train-val-test split
    ├── engine.py           # training loop (all optimizers) + accuracy/eval
    └── visualize.py        # digit + confusion matrix plotting
```

## Setup

```bash
git clone <your-repo-url>
cd digit-classification
pip install -r requirements.txt
```

Download `mnist_train_small.csv` and `mnist_test.csv` (e.g. from the
"MNIST in CSV" dataset on Kaggle) and place them in `data/`.

## Usage

```bash
python main.py
```

Optional flags:

```bash
python main.py --iterations 5000 --batch-size 64 --n-hidden 128
```

This will:
1. Load and split the data (train / 5000-example validation / remaining test)
2. Build a 4-hidden-layer MLP (`Linear -> BatchNorm1d -> PReLU`, x4)
3. Train it from the same initialization under RMSprop, Adam, AdaGrad,
   Momentum-SGD, and plain SGD
4. Print train/validation accuracy for each
5. Print test-set accuracy + confusion matrix for the RMSprop and plain-SGD
   models
6. Plot the two confusion matrices side by side

## Notes

- The model uses a custom Kaiming-style init scaled further by activation
  gain, plus a small final-layer scale-down for smoother early training.
- `BatchNorm1d.training` is toggled off automatically inside `calc_accuracy`
  so evaluation uses running statistics instead of batch statistics.
