import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importando las clases y funciones de los archivos
from layers import Layer_Dense
from activation_functions import softmax_train, relu, relu_derivative, cross_entropy_derivative
from optimizers import one_hot_encoding, training

EPOCH = 1200
LEARNING_RATE = 0.000001
np.random.seed(7)

# Prepare our data
data_set = pd.read_csv("data/mnist_train.csv")
X = data_set.iloc[0:, 1:].to_numpy() / 255  # Normalize
y_labels = data_set.iloc[:, 0:1].to_numpy().squeeze()
data_length = len(y_labels)

# Apply One Hot Encoding
y_train_labels = one_hot_encoding(y_labels, np.zeros((data_length, 10)))

# Create the model
hidden_layer = Layer_Dense(784, 32)
output_layer = Layer_Dense(32, 10)

# Training the model using the 'training' function
# Seleccionando el método deseado: 'SGD' o 'GD'
method = 'SGD'  # Cambiar a 'GD' si se desea utilizar Gradiente Descendente
hidden_layer, output_layer,losses = training(X, y_train_labels, hidden_layer, output_layer, LEARNING_RATE, EPOCH, method)

# Now let's test our model
# History
plt.plot(losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
