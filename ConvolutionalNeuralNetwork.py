import numpy as np
import pandas as pd
from Layer_Dense import Layer_Dense
import matplotlib.pyplot as plt

EPOCH = 20
LEARNING_RATE = .01

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Using np.random.randn for initialization and scaling the values
        self.weights = np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 
        return self.output
    
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

def softmax_train(x):
    x -= np.max(x , axis=1 , keepdims = True)
    exp_x = np.exp(x) #cada elemento de la matriz es exponente de e
    return exp_x / np.sum(exp_x, axis = 1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def cross_entropy_derivative(y_true,y_pred):
    return y_pred-y_true # -log(y_true/y_pred)    

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#Create the model

hidden_layer = Layer_Dense(
    image_length,
    32
)
output_layer = Layer_Dense(
    32,
    10
)
print(output_layer.weights)
print(hidden_layer.weights)
print(hidden_layer.biases)

#training function
print("START OF TRAINING")        
for it in range(EPOCH):
    #forward prop
    z1 = hidden_layer.forward(X) + hidden_layer.biases#(60000,32)
    a1 = relu(z1) 
    z2 = output_layer.forward(a1) #(60000,10)
    a2 = softmax_train(z2)
    #backward prop
    err1 = cross_entropy_derivative(y_train_labels , a2) #dE/da2 * da2/dz2
    err2 = (err1 @ output_layer.weights.T) * relu_derivative(z1) #dz2/da1 * da1/dz1
    #classic gradient descend method
    #weights
    output_layer.weights -= LEARNING_RATE * (a1.T @ err1)
    hidden_layer.weights -= LEARNING_RATE * (X.T @ err2)
    #bias
    hidden_layer.biases -= LEARNING_RATE * np.sum(err2, axis = 0 , keepdims=True)/data_length
    print("PROGRESS: "+"*"*it)       
print("TRAINING COMPLETE")

#Now Lets Test our 

print(output_layer.weights)
print(hidden_layer.weights)
print(hidden_layer.biases)

data_test=pd.read_csv("data/mnist_test.csv")
x_test=data_test.iloc[:,1:].to_numpy()/255
y_test=data_test.iloc[:,:1].to_numpy().squeeze()
print(x_test.shape)
print(y_test.shape)

def plot_misclassified(input_, pred, real):
    img = input_.reshape((28, 28))
    plt.title(f"Pred: {pred}, Real: {real}")
    plt.imshow(img, cmap="gray")
    plt.show()
    
def prediction(input_,y_true,n_cases):
    hits=0
    for i in range(n_cases):
        z1 = hidden_layer.forward(input_[i]) + hidden_layer.biases
        a1 = relu(z1)
        z2 = output_layer.forward(a1)
        a2 = softmax(z2)
        print(a2)
        pred = np.argmax(a2)
        print(f"Prediction: {pred} vs Real: {y_true[i]}",end=" ")
        if pred==y_true[i]:
            hits+=1
            print("CORRECT")
        else:
            plot_misclassified(input_[i], pred, y_true[i])  # Invocamos a la función para mostrar la imagen incorrectamente clasificada.
            print("INCORRECT")
    return "Precission: "+str(hits/n_cases)
print(prediction(x_test,y_test,100))