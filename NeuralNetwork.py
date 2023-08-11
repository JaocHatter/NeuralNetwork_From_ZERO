import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
LEARNING_RATE = 0.01
EPOCHS = 20
np.random.seed(7) #agregamos una semilla

df = pd.read_csv('data/mnist_train.csv')

imgs =df.iloc[0:,1:] #separamos las imagenes de los labels
X=imgs.to_numpy()
dataset_size=len(X)

y = df.iloc[:,0:1] #capturamos los labels
y = y.to_numpy()
unique, counts = np.unique(y, return_counts=True)

y_train = np.zeros((len(y), 10))
for i, y in enumerate(y):
    y_train[i][y] = 1

X=X/255
print(y_train.shape)
print(y_train[:5])
"""#Lets generate Data
def f(x,y):
    return 23.75*x-13.9*y+43.2
x_train=np.random.rand(1000,1)
y_train=np.random.rand(1000,1)

z_train=f(x_train+np.random.rand(1000,1)*0.1,y_train+np.random.rand(1000,1)*0.1)

X_input=np.column_stack((x_train,y_train)).T"""
#Activation functions

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_derivative(output):
    return output * (1 - output)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def cross_Entropy_grad(y_true,y_pred):
    return np.sum(-y_true/(y_pred+1e-15))

#Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_neurons,n_inputs) #W
        self.biases = np.zeros((n_neurons, 1))
    def forward(self, inputs):
        self.output = np.dot(self.weights,inputs) + self.biases

#Creating model
#2 hidden layers
layer1 = Layer_Dense(784, 32) 
layer2= Layer_Dense(32,16) 
output_layer=Layer_Dense(16,10)
#La salida es de 1output_layer.biases=np.zeros((10,1))
"""
        *   *
    *   
        *   *   *
    *           
        *   *
"""
#Training
for j in range(EPOCHS):
    for i in range(dataset_size):
        #Forward Propagation
        z_1=layer1.weights @ X[i].T + layer1.biases
        a_1=relu(z_1)
        z_2=layer2.weights @ a_1 + layer2.biases
        a_2=relu(z_2)
        z_3=output_layer.weights @ a_2
        a_3=softmax(z_3)
        #Computation of error
        loss= -np.sum(y_train[i]*np.log(a_3+1e-15))
        #Back Propagation
        delta_output=cross_Entropy_grad(y_train[i],a_3.T) * softmax_derivative(z_3)
        delta_layer2=(output_layer.weights.T * delta_output) * relu_derivative(z_2)
        delta_layer1=(layer2.weights.T * delta_layer2) * relu_derivative(z_1)
        #Updating parameters with Descent gradient algorithm
        #WEIGHTS
        output_layer.weights -= LEARNING_RATE * (delta_output*a_2.T)
        layer2.weights -= LEARNING_RATE * (delta_layer2*a_1.T)
        layer1.weights -= LEARNING_RATE * (delta_layer1*X[i].T)
        #BIAS
        layer2.biases -= LEARNING_RATE * delta_layer2
        layer1.biases -= LEARNING_RATE * delta_layer1
    print(f"epoch: {j} , loss: {loss} , process: "+"*"*(j/EPOCHS)*20)