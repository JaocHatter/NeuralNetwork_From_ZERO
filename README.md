# Neural Networks from Zero

Welcome to my first AI project! If you're here to explore how neural networks work, you're in the right place. In this walkthrough, you'll gain a deeper understanding of fundamental algorithms and theories in the Machine Learning using an simple example as follow...

## Simple Multilayer Neural Network

To elucidate concepts and theories, I've developed a neural network from the ground up. In other words, it's an artificial neural network functioning without relying on libraries such as TensorFlow or Keras. Instead, I've employed libraries like NumPy, Pandas, and Matplotlib to construct this network. This hands-on approach will give you insight into the core principles behind neural networks.

![mnist_plot](https://github.com/JaocHatter/NeuralNetwork_From_Zero/assets/112034917/613d66b1-db0b-49cb-9e5c-d1be4ff84679)

## Explanation
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
  Set the corresponding position in y_train to 1 for the label y_it<br>
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
Define the softmax function to compute the softmax activation
Define the softmax_derivative function to compute the derivative of softmax
Define the relu function to compute the ReLU activation
Define the relu_derivative function to compute the derivative of ReLU
Define the cross_entropy_derivative function to compute the derivative of cross-entropy loss
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
Define the class Layer_Dense:
Initialize with n_inputs and n_neurons
Initialize the weights as random values and biases as zeros
Define a forward method to compute the output of the layer
```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_neurons,n_inputs) #W
        self.biases = np.zeros((n_neurons, 1))
    def forward(self, inputs):
        self.output = np.dot(self.weights,inputs) + self.biases
```
### 9. Model Initialization <br>
Create instance of Layer_Dense called hidden_layer with n_inputs=784 and n_neurons=32
Create instance of Layer_Dense called output_layer with n_inputs=32 and n_neurons=10
```python
hidden_layer = Layer_Dense(n_inputs=784,n_neurons=32) 
output_layer=Layer_Dense(n_inputs=32,n_neurons=10)
```
### 10. Training Loop <br>
Loop over each epoch j in the range of EPOCHS:
Initialize total_loss to 0
Loop over each data point i in the range of 60000:
Perform forward propagation:
Compute z_1 using the hidden layer's weights and biases
Apply ReLU activation to get a_1
Compute z_2 using the output layer's weights
Apply softmax activation to get a_2
Compute the cross-entropy loss
Update total_loss with the calculated loss
Compute error terms for backpropagation
Update weights and biases using gradient descent
Calculate the mean loss for the epoch
Print epoch number, mean loss, and a process indicator
```python

```
### 11. Training Completion <br>
Print a message indicating that the neural network has been trained successfully
```python

```
### 12. Testing Data Preparation
Read the test data from the file "data/mnist_test.csv" into the DataFrame data_test
Extract test images and normalize them
Extract test labels and squeeze dimensions
```python

```
### 13. Prediction Function <br>
Define a function prediction that takes input_, y_true, and n_cases
Initialize hits to 0
Loop over each test case:
Perform forward propagation on the test data
Compare predicted label with true label and count hits
Print the prediction result (correct or incorrect)
Return the precision as the ratio of hits to the total number of test cases
### 14. Predictions and Evaluation
Call the prediction function with x_test, y_test, and 100 as arguments and print the precision
