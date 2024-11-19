# This code implements a simple perceptron for binary classification. 
# It trains on data with two features (A and B) using the perceptron learning rule, 
# adjusting weights based on prediction errors, and stops when all predictions are correct or after a set number of epochs.
#   w = w + epsilon * error * x_i

import numpy as np

# Training data
X = np.array([
    [1, 0, 0],  # x0=1, A=0, B=0
    [1, 0, 1],  # x0=1, A=0, B=1
    [1, 1, 0],  # x0=1, A=1, B=0
    [1, 1, 1]   # x0=1, A=1, B=1
])

# Target values
Y = np.array([0, 1, 1, 1])

# Initialize weights
w = np.array([0, 1, -1])

# Learning rate
epsilon = 1

# Activation function (threshold)
def activation_function(x):
    return 1 if x >= 0 else 0

# Perceptron training
def train_perceptron(X, Y, w, epsilon, max_epochs=1000):
    for epoch in range(max_epochs):
        for i in range(len(X)):
            x_i = X[i]
            y_i = Y[i]
            y_pred = activation_function(np.dot(w, x_i))
            error = y_i - y_pred
            w += epsilon * error * x_i
            if np.sum(np.abs(error)) == 0:
                return w  # Stop training if all predictions are correct
    return w

# Train the perceptron
trained_weights = train_perceptron(X, Y, w, epsilon)

# Display the trained weights
print("Trained Weights (w1, w2, w3):", trained_weights)