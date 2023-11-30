import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPOCH = 1200
LEARNING_RATE = 0.000001 #THIS NUMBER COULD BE YOUR WORST NIGHTMARE
np.random.seed(7) #agregamos una semilla

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs) #Kaiming Initialization or He Initialization!
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        #XW+b
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
    temp = x - np.max(x , axis=1 , keepdims = True)
    exp_x = np.exp(temp) #cada elemento de la matriz es exponente de e
    suma =  np.sum(exp_x, axis = 1, keepdims=True)
    return exp_x / suma

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1 , 0).astype(float)

def cross_entropy_derivative(y_true,y_pred):
    return y_pred-y_true # -log(y_true/y_pred)    

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#Create the model

hidden_layer = Layer_Dense(
    784,
    32
)
output_layer = Layer_Dense(
    32, 
    10
)

#training function using classic gradient descend method
print("START OF TRAINING")        
losses = []
for it in range(EPOCH):
    #forward prop
    z1 = hidden_layer.forward(X) #(60000,32)
    a1 = relu(z1) 
    z2 = output_layer.forward(a1) #(60000,10)
    a2 = softmax_train(z2)
    #computing the loss
    loss = -np.sum(y_train_labels * np.log(a2+1e-10)) / data_length
    losses.append(loss)
    #backward prop
    #Output layer
    err1 = cross_entropy_derivative(y_train_labels , a2) #dE/da2 * da2/dz2
    output_layer.weights -= LEARNING_RATE * (a1.T @ err1)
    output_layer.biases -= LEARNING_RATE * np.sum(err1, axis=0 , keepdims=True)
    #Hidden layer
    err2 = (err1 @ output_layer.weights.T) * relu_derivative(z1) #dz2/da1 * da1/dz1
    hidden_layer.weights -= LEARNING_RATE * (X.T @ err2)
    hidden_layer.biases -= LEARNING_RATE * np.sum(err2, axis = 0 , keepdims=True)
    print(f"epoch: {it+1} , loss: {loss} , process: "+"*"*(int((it/EPOCH)*20)))
print("TRAINING COMPLETE")

#History 

plt.plot(losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#Now Lets Test our model

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
        z1 = hidden_layer.forward(input_[i]) 
        a1 = relu(z1)
        z2 = output_layer.forward(a1) 
        a2 = softmax(z2)
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