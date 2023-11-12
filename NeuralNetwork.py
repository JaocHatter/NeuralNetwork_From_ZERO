import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from numba import jit,njit

LEARNING_RATE = 0.01
EPOCHS = 5
np.random.seed(7)  # agregamos una semilla

df = pl.read_csv("data/mnist_train.csv")

#imgs = df.iloc[0:, 1:]  # separamos las imagenes de los labels
X = df.drop('label').to_numpy().T
y = df.select('label').to_numpy().squeeze()

unique, counts = np.unique(y, return_counts=True)

y_train = np.zeros((10, len(y)))
@njit
def one_hot_encoding(y_in, y_out):
    for i, y_it in enumerate(y_in):
        y_out[y_it][i] = 1
    return y_out

y_train = one_hot_encoding(y, y_train)

# Normalizing data to prevent overflow
X = X / 255

# Activation functions

@jit(nopython = True)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@jit(nopython = True)
def softmax_derivative(output):
    return output * (1 - output)

@jit(nopython = True)
def relu(x):
    return np.maximum(0, x)

@jit(nopython = True)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

@jit(nopython=True)
def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true  # -log(y_true/y_pred)


# Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_neurons, n_inputs)  # W
        self.biases = np.zeros((n_neurons, 1))

    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.biases


# Creating model
# 2 hidden layers
hidden_layer = Layer_Dense(n_inputs=784, n_neurons=32)
output_layer = Layer_Dense(n_inputs=32, n_neurons=10)
print("training in process...")
# Training #MODIFY!

def training(X, y_train, hidden_layer, output_layer,epochs):
    for j in range(epochs):
        total_loss = 0
        for i in range(60000):
            # Forward Propagation
            z_1 = hidden_layer.weights @ X[:, i : i + 1] + hidden_layer.biases
            a_1 = relu(z_1)
            z_2 = output_layer.weights @ a_1
            a_2 = softmax(z_2)
            # Computation of error
            loss = -np.sum(
                y_train[:, i : i + 1] * np.log(a_2 + 1e-10)
            )  # Adding epsilon to avoid log(0)
            total_loss += loss
            # Back Propagation
            error_term_output = cross_entropy_derivative(
                y_train[:, i : i + 1], a_2
            )  # dL/dA_2 * dA_2/dZ_2
            error_term_hidden = (
                output_layer.weights.T @ error_term_output
            ) * relu_derivative(z_1)
            # Updating parameters
            # WEIGHTS
            output_layer.weights -= LEARNING_RATE * (
                error_term_output @ a_1.T
            )  # delta @ dZ_2/dW_
            hidden_layer.weights -= LEARNING_RATE * (error_term_hidden @ X[:, i : i + 1].T)

            # BIAS
            hidden_layer.biases -= LEARNING_RATE * error_term_hidden
        mean_loss = total_loss / 60000
        print(
            f"epoch: {j+1} , loss: {mean_loss} , process: " + "*" * (int((j / epochs) * 20))
        )
    return hidden_layer,output_layer
#Train and update neural network
  
hidden_layer,output_layer = training(X, y_train, hidden_layer, output_layer,epochs=EPOCHS)
print("Neural network trained sucessfully!")

#Now test our model with new data

data_test = pl.read_csv("data/mnist_test.csv")
x_test = data_test.drop('label').to_numpy().T / 255
y_test = data_test.select('label').to_numpy().T.squeeze()
print(x_test.shape)
print(y_test.shape)

def plot_misclassified(input_, pred, real):
    img = input_.reshape((28, 28))
    plt.title(f"Pred: {pred}, Real: {real}")
    plt.imshow(img, cmap="gray")
    plt.show()
    
def prediction(input_, y_true, n_cases):
    hits = 0
    for i in range(n_cases):
        z1 = hidden_layer.weights @ input_[:, i : i + 1] + hidden_layer.biases
        a1 = relu(z1)
        z2 = output_layer.weights @ a1
        a2 = softmax(z2).T
        pred = np.argmax(a2)
        print(f"Prediction: {pred} vs Real: {y_true[i]}", end=" ")
        if pred == y_true[i]:
            hits += 1
            print("CORRECT")
        else:
            plot_misclassified(input_[:, i], pred, y_true[i])
            print("INCORRECT")
    return "Precission: " + str(hits / n_cases)

print(prediction(x_test, y_test, 100))
