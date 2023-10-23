import numpy as np
import pandas as pd
from Layer_Dense import Layer_Dense

EPOCH = 4
LEARNING_RATE = .05
#Prepare our data

data_set=pd.read_csv("data/mnist_train.csv")
X = data_set.iloc[0:,1:].to_numpy()/255 #Normalize
y_labels = data_set.iloc[:,0:1].to_numpy().squeeze()
data_length = len(y_labels)

image_length = len(X[1])

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

hidden_layer = Layer_Dense(
    image_length,
    32
)
output_layer = Layer_Dense(
    32,
    10
)

#training function

def train_model(x_,y_):
    #Calculate according our theory
    print("START OF TRAINING")        
    for it in range(EPOCH):
        #forward prop
        z1 = hidden_layer.forward(x_) #(60000,32)
        a1 = relu(z1) 
        z2 = output_layer.forward(a1) #(60000,10)
        a2 = softmax(z2)
        #backward prop
        err1 = cross_entropy_derivative(y_ , a2) #dE/da2 * da2/dz2
        err2 = (err1 @ hidden_layer.weights) * relu_derivative(z1) #dz2/da1 * da1/dz1
        #gradient descend
        #weights
        output_layer.weights -= LEARNING_RATE * (a1.T @ err1)
        hidden_layer.weights -= LEARNING_RATE * (x_.T @ err2)
        #bias
        output_layer.biases -= LEARNING_RATE * err1
        hidden_layer.biases -= LEARNING_RATE * err2
        print("PROGRESS: "+"*"*it)       
    print("TRAINING COMPLETE")

#training my model