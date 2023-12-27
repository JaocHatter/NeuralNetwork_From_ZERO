import numpy as np
from activation_functions import softmax,relu,relu_derivative,cross_entropy_derivative,softmax_derivative
activation = {"relu": relu , "softmax" : softmax}
derivative = {"relu": relu_derivative, "softmax": softmax_derivative, "categorical_crossentropy":cross_entropy_derivative}
class Dense:
    """
    Clase para representar una capa densa en una red neuronal.

    Atributos:
        n_inputs (int): Número de entradas.
        n_neurons (int): Número de neuronas.
        weights (ndarray): Pesos de la capa.
        biases (ndarray): Sesgos de la capa.
        output (ndarray): Salida de la capa.
    """

    def __init__(self,n_neurons,activation,input_shape):
        """
        Inicializa la capa densa con pesos y sesgos aleatorios.

        Parámetros:
            n_inputs (int): Número de entradas.
            n_neurons (int): Número de neuronas.
        """
        self.n_neurons = n_neurons
        self.activation = activation
        self.input_shape = input_shape
        self.weights = np.random.randn(input_shape, n_neurons) * np.sqrt(2. / self.input_shape)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        """
        Realiza la operación de forward pass en la capa.

        Parámetros:
            inputs (ndarray): Entradas de la capa.
        """
        self.output = activation[self.activation](inputs @ self.weights + self.biases)
        return self.output
    def backward(self):
        return
