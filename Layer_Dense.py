import numpy as np
class Layer_Dense:
    """
    Clase para representar una capa densa en una red neuronal.

    Atributos:
        n_inputs (int): Número de entradas.
        n_neurons (int): Número de neuronas.
        weights (ndarray): Pesos de la capa.
        biases (ndarray): Sesgos de la capa.
        output (ndarray): Salida de la capa.
    """

    def __init__(self, n_inputs, n_neurons):
        """
        Inicializa la capa densa con pesos y sesgos aleatorios.

        Parámetros:
            n_inputs (int): Número de entradas.
            n_neurons (int): Número de neuronas.
        """
        self.weights = np.random.rand(n_neurons, n_inputs)
        self.biases = np.zeros((n_neurons, 1))

    def forward(self, inputs):
        """
        Realiza la operación de forward pass en la capa.

        Parámetros:
            inputs (ndarray): Entradas de la capa.
        """
        self.output = np.dot(self.weights, inputs) + self.biases
