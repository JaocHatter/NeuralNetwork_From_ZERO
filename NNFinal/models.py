from activation_functions import *
from layers import *
class Sequential:
    def __init__(self):
        self.layers = []
        self.outputs = []
        self.optimizer = None
        self.loss_function = None
        self.epochs = 10
        self.accuracy = False
    def add(self,layer):
        self.layers.append(layer)
    def compile(self,optimizer,loss_function,accuracy,lr):
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.accuracy = accuracy
        self.learning_rate = lr
    def fit(self,features,labels,epochs):
        self.epochs = epochs
        self.n_layers = len(self.layers)
        self.outputs=[features]
        print(self.outputs[0].shape)
        """
        Entrena la red neuronal usando Gradiente Descendente o Gradiente Descendente Estocástico.

        Parámetros:
            X (ndarray): Datos de entrenamiento.
            y_train (ndarray): Etiquetas de entrenamiento en formato one-hot.
            hidden_layer (Layer_Dense): Capa oculta de la red.
            output_layer (Layer_Dense): Capa de salida de la red.
            epochs (int): Número de épocas de entrenamiento.
            method (str): Método de entrenamiento, 'SGD' o 'GD'.

        Retorna:
            tuple: Capa oculta y capa de salida actualizadas después del entrenamiento.
        """
        data_lenght = features.shape[0]
        self.losses = []
        if self.optimizer == 'GD':
            for epoch in range(self.epochs):
                #fordward prop
                for idx in range(0,self.n_layers):
                    self.outputs.append(self.layers[idx].forward((self.outputs)[idx]))
                loss = -np.sum(labels * np.log(self.outputs[-1]+1e-10)) / data_lenght
                self.losses.append(loss)
                #backward prop
                error = derivative[self.loss_function](labels,self.outputs[-1]) # dE/dA(l)*dA(l)/dZ(l)
                for idx in range(self.n_layers-1,-1,-1):
                    self.layers[idx].weights -= self.learning_rate * (self.outputs[idx].T @ error) # W(l) = W(l) - lr * e @ dZ(l)/dW(l)
                    self.layers[idx].biases -= self.learning_rate * np.sum(error, axis = 0 , keepdims=True)
                    if idx != 0:
                        error = ( error @ self.layers[idx].weights.T) * derivative[self.layers[idx-1].activation]( self.outputs[idx] ) 
                #Restablecemos nuestras salidas
                self.outputs = [features]
                print(f"epoch: {epoch+1} , loss: {loss} , process: "+"*"*(int((epoch/self.epochs)*20)))
            print("TRAINING COMPLETE")
        #manhana sin falta carnal
        elif self.optimizer == 'SGD':
            for epoch in range(self.epochs):
                total_loss = 0
                for i in range():
                    return  
        return
    def predict(self,x_test,y_test):
        #fordward prop nms
        for idx in range(1,self.n_layers):
            self.layers[idx].forward(self.layers[idx-1].output)
        return np.argmax(self.layers[-1].output)