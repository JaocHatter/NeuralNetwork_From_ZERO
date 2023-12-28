from activation_functions import *
from layers import *

class Sequential:
    """
    Clase que representa un modelo de red neuronal secuencial.

    Atributos:
        layers (list): Lista de capas en el modelo.
        outputs (list): Lista que almacena las salidas de cada capa durante la propagación hacia adelante.
        optimizer (str): Optimizador a utilizar para el entrenamiento ('GD' para Gradiente Descendente, 'SGD' para Gradiente Descendente Estocástico).
        loss_function (str): Función de pérdida a utilizar durante el entrenamiento.
        epochs (int): Número de épocas de entrenamiento.
        accuracy (bool): Indica si se debe calcular la precisión durante el entrenamiento.
        learning_rate (float): Tasa de aprendizaje para el entrenamiento.
        losses (list): Lista que almacena las pérdidas durante el entrenamiento.
    """

    def __init__(self):
        """
        Inicializa un modelo secuencial.
        """
        self.layers = []
        self.outputs = []
        self.optimizer = None
        self.loss_function = None
        self.epochs = 10
        self.accuracy = False
        self.learning_rate = None
        self.losses = []

    def add(self, layer):
        """
        Agrega una capa al modelo.

        Parámetros:
            layer (Layer): Capa a agregar al modelo.
        """
        self.layers.append(layer)

    def compile(self, optimizer, loss_function, accuracy, lr):
        """
        Compila el modelo con configuraciones específicas para el entrenamiento.

        Parámetros:
            optimizer (str): Optimizador a utilizar para el entrenamiento ('GD' o 'SGD').
            loss_function (str): Función de pérdida a utilizar durante el entrenamiento.
            accuracy (bool): Indica si se debe calcular la precisión durante el entrenamiento.
            lr (float): Tasa de aprendizaje para el entrenamiento.
        """
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.accuracy = accuracy
        self.learning_rate = lr

    def fit(self, features, labels, epochs):
        """
        Entrena la red neuronal usando Gradiente Descendente o Gradiente Descendente Estocástico.

        Parámetros:
            features (ndarray): Datos de entrenamiento.
            labels (ndarray): Etiquetas de entrenamiento en formato one-hot.
            epochs (int): Número de épocas de entrenamiento.
        """
        self.epochs = epochs
        self.n_layers = len(self.layers)
        self.outputs = [features]

        data_length = features.shape[0]
        self.losses = []

        if self.optimizer == 'GD':
            for epoch in range(self.epochs):
                # Propagación hacia adelante (forward propagation)
                for idx in range(0, self.n_layers):
                    self.outputs.append(self.layers[idx].forward((self.outputs)[idx]))

                # Cálculo de la pérdida
                loss = -np.sum(labels * np.log(self.outputs[-1] + 1e-10)) / data_length
                self.losses.append(loss)

                # Retropropagación (backward propagation)
                error = derivative[self.loss_function](labels, self.outputs[-1])
                for idx in range(self.n_layers - 1, -1, -1):
                    self.layers[idx].weights -= self.learning_rate * (self.outputs[idx].T @ error)
                    self.layers[idx].biases -= self.learning_rate * np.sum(error, axis=0, keepdims=True)
                    if idx != 0:
                        error = (error @ self.layers[idx].weights.T) * derivative[self.layers[idx - 1].activation](
                            self.outputs[idx]
                        )

                # Restablecemos nuestras salidas
                self.outputs = [features]
                print(f"epoch: {epoch + 1} , loss: {loss} , process: " + "*" * (int((epoch / self.epochs) * 20)))
            print("TRAINING COMPLETE")

        elif self.optimizer == 'SGD':
            total_loss = 0
            self.layers[-1].activation = "softmax_sgd"
            for epoch in range(self.epochs):
                total_loss = 0
                for i in range(data_length):
                    self.outputs = [features[i:i + 1, :]]
                    for idx in range(0, self.n_layers):
                        self.outputs.append(self.layers[idx].forward((self.outputs)[idx]))

                    loss = -np.sum(labels[i:i + 1, :] * np.log(self.outputs[-1] + 1e-10))
                    total_loss += loss

                    # Retropropagación (backward propagation)
                    error = derivative[self.loss_function](labels[i:i + 1, :], self.outputs[-1])
                    for idx in range(self.n_layers - 1, -1, -1):
                        self.layers[idx].weights -= self.learning_rate * (self.outputs[idx].T @ error)
                        self.layers[idx].biases -= self.learning_rate * np.sum(error, axis=0, keepdims=True)
                        if idx != 0:
                            error = (error @ self.layers[idx].weights.T) * derivative[self.layers[idx - 1].activation](
                                self.outputs[idx]
                            )

                mean_loss = total_loss / data_length
                self.losses.append(mean_loss)
                print(
                    f"epoch: {epoch + 1} , loss: {mean_loss} , process: " + "*" * (int((epoch / self.epochs) * 20))
                )

    def predict(self, x_test, y_test):
        """
        Realiza predicciones utilizando la red neuronal entrenada.

        Parámetros:
            x_test (ndarray): Datos de prueba.
            y_test (ndarray): Etiquetas de prueba en formato one-hot.

        Retorna:
            int: Índice de la clase predicha.
        """
        # Propagación hacia adelante (forward propagation)
        for idx in range(1, self.n_layers):
            self.layers[idx].forward(self.layers[idx - 1].output)
        return np.argmax(self.layers[-1].output)
