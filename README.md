# Neural Networks for Newbies

Welcome to my first AI project! If you're here to explore how neural networks work, you're in the right place. In this walkthrough, you'll gain a deeper understanding of fundamental algorithms and theories in the Machine Learning using an simple example as follow...

## Simple Multilayer Neural Network

To elucidate concepts and theories, I've developed a neural network from the ground up. In other words, it's an artificial neural network functioning without relying on libraries such as TensorFlow or Keras. Instead, I've employed libraries like NumPy, Pandas, and Matplotlib to construct this network. This hands-on approach will give you insight into the core principles behind neural networks.

![mnist_plot](https://github.com/JaocHatter/NeuralNetwork_From_Zero/assets/112034917/613d66b1-db0b-49cb-9e5c-d1be4ff84679)
## Dataset
You can download it in [Link Text](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) 
## Explanation
In this explanation i will use the first code i made in this repository, a training that uses Stochastic Gradient Descent
### 1. Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
### 2. Constants and Seed <br>
```python
LEARNING_RATE = 0.01
EPOCHS = 20
np.random.seed(7) #agregamos una semilla
df = pd.read_csv('data/mnist_train.csv')

```
### 3. Loading Data <br>
Read the CSV data from the file 'data/mnist_train.csv' into the DataFrame df
Note: all data sets are saved in the "data" folder... you must get it using the following link
```plaintext
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
```
```python
df = pd.read_csv('data/mnist_train.csv')
```
### 4. Data Preparation <br>
Extract images data by selecting all rows of df except the first column, stored in imgs<br>
Convert imgs to a numpy array and transpose it to get X<br>
Extract labels by selecting only the first column of df, stored in y<br>
Squeeze the dimensions of y to remove unnecessary dimensions<br>
Calculate unique values and their counts in y, stored in unique and counts respectively<br>
```python
imgs =df.iloc[0:,1:] #separamos las imagenes de los labels
X=imgs.to_numpy().T
y = df.iloc[:,0:1] #capturamos los labels
y = y.to_numpy().squeeze()
unique, counts = np.unique(y, return_counts=True)
```
### 5. One-Hot Encoding Labels <br>
Create an array y_train of zeros with a shape of (10, length of y)<br>
Loop over each label y_it and corresponding index i in y:<br>
&emsp;Set the corresponding position in y_train to 1 for the label y_it<br>
```python
y_train = np.zeros((10,len(y)))
for i, y_it in enumerate(y):
    y_train[y_it][i] = 1
```
### 6. Data Normalization <br>
Normalize the X array by dividing it element-wise by 255
```python
X=X/255
```
### 7.  Activation Functions <br>
Define the softmax function to compute the softmax activation <br>
Define the softmax_derivative function to compute the derivative of softmax<br>
Define the relu function to compute the ReLU activation<br>
Define the relu_derivative function to compute the derivative of ReLU<br>
Define the cross_entropy_derivative function to compute the derivative of cross-entropy loss<br>
```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def softmax_derivative(output):
    return output * (1 - output)
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
def cross_entropy_derivative(y_true,y_pred):
    return y_pred-y_true # -log(y_true/y_pred)    
```
### 8. Layer Definition <br>
Define the class Layer_Dense:<br>
&emsp;Initialize with n_inputs and n_neurons<br>
&emsp;Initialize the weights as random values and biases as zeros<br>
&emsp;Define a forward method to compute the output of the layer<br>
```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_neurons,n_inputs) #W
        self.biases = np.zeros((n_neurons, 1))
    def forward(self, inputs):
        self.output = np.dot(self.weights,inputs) + self.biases
```
### 9. Model Initialization <br>
Create instance of Layer_Dense called hidden_layer with n_inputs=784 and n_neurons=32<br>
Create instance of Layer_Dense called output_layer with n_inputs=32 and n_neurons=10<br>
```python
hidden_layer = Layer_Dense(n_inputs=784,n_neurons=32) 
output_layer=Layer_Dense(n_inputs=32,n_neurons=10)
```
### 10. Training Loop <br>
Loop over each epoch j in the range of EPOCHS:<br>
&emsp;Initialize total_loss to 0<br>
&emsp;Loop over each data point i in the range of 60000:<br>
&emsp;&emsp;Perform forward propagation:<br>
&emsp;&emsp;Compute z_1 (first layer output) using the hidden layer's weights and biases<br>
&emsp;&emsp;Apply ReLU activation to get a_1<br>
&emsp;&emsp;Compute z_2 (second layer output) using the output layer's weights<br>
&emsp;&emsp;Apply softmax activation to get a_2<br>
&emsp;&emsp;Compute the cross-entropy loss<br>
&emsp;&emsp;Update total_loss with the calculated loss<br>
&emsp;&emsp;Compute error terms for backpropagation<br>
&emsp;&emsp;Update weights and biases using gradient descent<br>
&emsp;Calculate the mean loss for the epoch<br>
Print epoch number, mean loss, and a process indicator<br>
```python
for j in range(EPOCHS):
    total_loss=0
    for i in range(60000):
        #Forward Propagation
        z_1=hidden_layer.weights @ X[:,i:i+1] + hidden_layer.biases
        a_1=relu(z_1)
        z_2=output_layer.weights @ a_1
        a_2=softmax(z_2)
        #Computation of error
        loss = -np.sum(y_train[:, i:i+1] * np.log(a_2 + 1e-10))  # Adding epsilon to avoid log(0)
        total_loss+=loss
        #Back Propagation
        error_term_output = cross_entropy_derivative(y_train[:,i:i+1],a_2) #dL/dA_2 * dA_2/dZ_2
        error_term_hidden = (output_layer.weights.T @ error_term_output) * relu_derivative(z_1)
        #Updating parameters 
        #WEIGHTS
        output_layer.weights -= LEARNING_RATE * (error_term_output @ a_1.T) #delta @ dZ_2/dW_
        hidden_layer.weights -= LEARNING_RATE * (error_term_hidden @ X[:,i:i+1].T)
        
        #BIAS
        hidden_layer.biases -= LEARNING_RATE * error_term_hidden
    mean_loss = total_loss/60000
    print(f"epoch: {j+1} , loss: {mean_loss} , process: "+"*"*(int((j/EPOCHS)*20)))  
```
### 11. Training Completion <br>
Print a message indicating that the neural network has been trained successfully<br>
### 12. Testing Data Preparation
Read the test data from the file "data/mnist_test.csv" into the DataFrame data_test<br>
Extract test images and normalize them<br>
Extract test labels and squeeze dimensions<br>
```python
data_test=pd.read_csv("data/mnist_test.csv")
x_test=data_test.iloc[:,1:].to_numpy().T/255
y_test=data_test.iloc[:,:1].to_numpy().T.squeeze()
```
### 13. Prediction Function <br>
Define a function prediction that takes input_, y_true, and n_cases<br>
Initialize hits to 0<br>
Loop over each test case:<br>
&emsp;Perform forward propagation on the test data<br>
&emsp;Compare predicted label with true label and count hits<br>
&emsp;Print the prediction result (correct or incorrect)<br>
Return the precision as the ratio of hits to the total number of test cases<br>
```python
def prediction(input_,y_true,n_cases):
    hits=0
    for i in range(n_cases):
        z1=hidden_layer.weights @ input_[:,i:i+1] + hidden_layer.biases
        a1=relu(z1)
        z2=output_layer.weights @ a1
        a2=softmax(z2).T
        pred=np.argmax(a2)
        print(f"Prediction: {pred} vs Real: {y_true[i]}",end=" ")
        if pred==y_true[i]:
            hits+=1
            print("CORRECT")
        else:
            print("INCORRECT")
    return "Precission: "+str(hits/n_cases)
```
### 14. Predictions and Evaluation<br>
Call the prediction function with x_test, y_test, and 100 as arguments and print the precision<br>
```python
print(prediction(x_test,y_test,100))
```
