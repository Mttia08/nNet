"""
This is a python package for building neural networks 
using only Dense (Fully Conected) Layers. It can be used
for various Regression tasks.

Usage: 
Put "from nNet import *" or "import nNet" at the top of your code 
and then create an instance of the Model class.
Eg. "model = Model(layer_sizes=[10, 20, 10], activations=["relu", "sigmoid", None], learning_rate=0.01)"
layer_sizes is the amount of layers and neurons per layer. In the previous example, 
the neural network will have 3 layers with 10, 20 and 10 neurons.
The activations specify each activation function used in each layer. 
It needs to have the same length as the layer_size parameter, if you
want a layer to have no activation function, use None.
The learning_rate specifies the learning rate of the model.

The train function will allow you to train the neural network,
you can to specify x and y, the epochs as well as whether to show
the current epoch and and error.
Eg. "model.train(x, y, epochs=10, showTraining=False)"

You can use the predict function, that takes an x as either a NumPy array 
or a single number, to predict the corresponding y value.
Eg. "prediction = model.predict(x)"

You can use the save function to save your model to a json file.
Eg. model.save("model.json")

There is also the load model function, which loads a model from
a json file to a variable.
Eg. model = load_model("model.json")
"""

import numpy as np
import json

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)

def gaussian(x):
    return np.exp(-x**2)

def softplus(x):
    return np.log(1 + np.exp(x))

activation_functions = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "swish": swish,
    "gaussian": gaussian,
    "softplus": softplus,
    None: lambda x: x
}

class Model():
    def __init__(self, layer_sizes, activations=None, learning_rate=0.01):
        if activations is None:
            self.activations = [None for _ in layer_sizes]
        else:
            if len(activations) != len(layer_sizes):
                raise ValueError("Length of activations does not match length of layer_sizes")
            self.activations = [activation_functions[act] for act in activations]
            self.act_str = activations
            
        self.learning_rate = learning_rate
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def train(self, X, y, epochs=10, showTraining=True):
        for epoch in range(1, epochs+1):
            for input_X, input_y in zip(X, y):
                prediction = self.forward(input_X)
                error = prediction - input_y
                self.backward(input_X, error)
            if showTraining:
                print(f"Epoch {epoch} ---------- Error: {np.abs(error).mean()}")

    def forward(self, X):
        self.a = [X]
        for i, weight in enumerate(self.weights):
            X = np.dot(X, weight)
            if self.activations[i]:
                X = self.activations[i](X)
            self.a.append(X)
        return X


    def backward(self, X, error):
        for i in reversed(range(len(self.weights))):
            gradient = np.outer(self.a[i], error)
            error = np.dot(error, self.weights[i].T)
            self.weights[i] -= self.learning_rate * gradient

    def predict(self, X):
        output = [self.forward(val) for val in X]
        return output

    def save(self, path):
        weights_as_lists = [w.tolist() for w in self.weights]
        with open(path, "w") as fp:
            json.dump({"activations": self.act_str, "weights": weights_as_lists}, fp)

def load_model(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    act_str = data["activations"]
    weights = [np.array(w) for w in data["weights"]]
    layer_sizes = [w.shape[0] for w in weights]
    model = Model(layer_sizes=layer_sizes, activations=act_str)
    model.weights = weights
    return model
