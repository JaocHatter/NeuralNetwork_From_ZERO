import numpy as np
from activation_functions import softmax,softmax_train,relu,relu_derivative,cross_entropy_derivative
def one_hot_encoding(y_in, y_out):
    """
    Convierte etiquetas en representación one-hot.

    Parámetros:
        y_in (ndarray): Vector de etiquetas.
        y_out (ndarray): Matriz de salida para la codificación one-hot.

    Retorna:
        ndarray: Matriz con codificación one-hot.
    """
    for i in range(len(y_in)):
        y_out[i][y_in[i]]=1
    return y_out
