import numpy as np
import pandas as pd
#Define our Layer_Dense Class

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.rand(n_inputs,n_neurons) #W
        self.biases = np.zeros((n_neurons, 1))
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.biases # XW + b

#Prepare our data

data_set=pd.read_csv("data/mnist_train.csv")
X = data_set.iloc[0:,1:].to_numpy()/255 #Normalize
y_labels = data_set.iloc[:,0:1].to_numpy().squeeze()
data_length = len(y_labels)

#Apply One Hot Encodding

y_train_labels = np.zeros((data_length,10))

for i in range(data_length):
    y_train_labels[i][y_labels[i]]=1

#Define our activation function, loss function , derivatives, etc...

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def cross_entropy_derivative(y_true,y_pred):
    return y_pred-y_true # -log(y_true/y_pred)    

#Create the model

