# Digit-Classifier-MLP
A personal project for better learning and understanding. While I did watch and took inspiration from Andrej Karpathy's "Neural Networks : Zero to hero" series, I did code it all after understanding the concepts, not while watching. No hand-derived backward function, used autograd.

## What does it do

As you might have guessed from the title, it classifies handwritten digits (0-9) from the MNIST dataset using multilayer perceptron (MLP) built from scratch.

## Dataset

- **MNIST** — 28x28 grayscale images of handwritten digits
- I have uploaded both the test and train dataset .csv files here.

### Split

- Training examples : 19999 (66.67%)
- Evaluation examples : 5000 (16.67%)
- Testing examples : 4999 (16.66%)

## Architecture

Although you can play with it, here's what I used

- Input layer: 784 (28x28 flattened)
- Hidden layer(s): all hidden layers have 260 neurons [Figured out from training multiple times]
- Output layer: 10 (digits 0–9)
- Activation: PReLU on hidden layers and SoftMax at the end
- Loss function: cross-entropy loss
- Optimizer: RMSprop

## Hyperparameters
Iterations : 14000
Batch size : 32
Learning rate for RMSprop : 0.001
Decay rate for RMSprop (γ) : 0.9 
ε for RMSprop : 1e-5
Learning rate for Batch Gradient Descent : 0.1
L2 Regularization parameter (α) : 0.01

## My results

- Training accuracy   : 99.94%
- Evaluation accuracy : 96.90%
- Test accuracy       : 98.68%

I do have doubts about these accuracy scores. Any feedback on this is welcome.

Although, I got an even higher training loss on Batch Gradient Descent optimized model.

- Training accuracy   : 99.97%
- Evaluation accuracy : 96.78%
- Test accuracy       : 98.78%

Seems unusual to me but I don't know how to explain it yet. 

## Confusion matrices
Brighter boxes denote higher error rate.

### For Batch Gradient Descent optimized model
We see highest confusion for classifying 4/9 and 3/5

<img width="312" height="201" alt="image" src="https://github.com/user-attachments/assets/011b9a10-5450-4ae3-9c74-5622d85aa783" />

### For RMSprop optimized model
We see highest confusion for classifying 4/9 while it maybe has confusion between 1s and 2s at times

<img width="309" height="205" alt="image" src="https://github.com/user-attachments/assets/b5ef599a-6657-4da3-8e85-5d18cd8c749a" />

## Explanation for Topics mentioned

### Multilayer Perceptron
A perceptron refers to a node that takes in multiple values as a vector of *d* dimensions (supposing there are *d* input values), computes a dot product with a weights vector of same dimensions, adds a bias term and passes the whole result through a non-linear function (activation function) to compute the output for that node.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/8828dedd-b206-4365-8f78-83b13a8c576b" />

And now just stack all these nodes on top of each other and compute outputs for each node, each having separate weights and biases. Each node output being passed through the activation function. And we have a single-layered perceptron.

<img width="600" height="478" alt="image" src="https://github.com/user-attachments/assets/60a6013b-dd4e-41da-88b4-d3498bbe59e8" />

Then arrange many such layers such that the outputs of the previous layer go in as the inputs to the next layer and we have a multilayered perceptron.

<img width="800" height="358" alt="image" src="https://github.com/user-attachments/assets/2a2cd365-7a5c-44aa-827e-31ce21283261" />

We can achieve various results through these architectures such as langauge modelling, classification, both on images or plane data, pattern recognition and more.

### RMSprop optimizer
Now, the image might look complicated but it's simple.
We keep track of the running averages of the squared gradients for various parameters our network has. 
Then simply calculate the update step as the gradient divided by the square root of this running mean plus a small constant used for preventing division by zero.

In my code, the β is written as γ. It represents the same quantity.

<img width="587" height="496" alt="image" src="https://github.com/user-attachments/assets/cbbd9a2c-8a4e-44a3-a6f1-fc1f436297af" />

### PReLU activation function

It's a ReLU function but with a learnable parameter for when the input is less than 0

*a* is a learnable parameter here

<img width="443" height="408" alt="image" src="https://github.com/user-attachments/assets/91e40075-bc83-4f43-a4bf-39197416bfe5" />

### SoftMax activation function

It transforms a layer's outputs such that sum of all the outputs is 1. It can be used to convert the outputs to probabilities for classification problems.

<img width="772" height="397" alt="image" src="https://github.com/user-attachments/assets/2bb49aad-99d6-4e9b-906b-d24f338202df" />

### Others...
Covering all of the optimizers would be pretty lengthy and hard to read. This code is free to fiddle with and I have added Adam, AdaGrad and Momentum based Batch Gradient Descent into it so one could experiment freely.

There are also various regularizers available such as Lasso (L1) or Ridge (L2) regularizers and dropout layers as well for further experimentation.

## Acknowledgments

- I am uploading something for the first time here so feedback on practices or code structure is welcome.
- Inspired by Andrej Karpathy's neural networks series.

##  Improvements in future
- Handwritten backward function.
- Adding even more optimizers and regularizers.
