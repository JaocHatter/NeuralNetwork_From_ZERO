# Neural Networks from zero

Welcome to my first AI project! If you're here to explore how neural networks work, you're in the right place. 
In this walkthrough, you'll gain a deeper understanding of fundamental algorithms and theories in the realm of Machine Learning.

## Simple Multilayer Neural Network

To elucidate concepts and theories, I've developed a neural network from the ground up. 
In other words, it's an artificial neural network functioning without relying on libraries such as TensorFlow or Keras. 
Instead, I've employed libraries like NumPy, Pandas, and Matplotlib to construct this network. 
This hands-on approach will give you insight into the core principles behind neural networks.

## Pseudocode

Constant "LEARNING_RATE" = 0.01
Constant "EPOCHS" = 20

np.random.seed(7)

"df" = Read CSV from file "data/mnist_train.csv"

"imgs" = Select all rows from "df" except the first column
"X" = Convert "imgs" to a numpy array and transpose it
"y" = Select only the first column from "df"
"y" = Squeeze "y" to remove dimensions
Compute unique values and their counts in "y"

"y_train" = Create a zeros array of shape (10, length of "y")
For each "i" and corresponding "y_it" in "y":
    Set "y_train[y_it][i]" to 1

Normalize "X" by dividing it element-wise by 255

Define "softmax" function that takes "x":
    Compute "e_x" as exponentiation of "x" minus maximum of "x"
    Return exponentiation of "x" divided by sum of "e_x"

Define "softmax_derivative" function that takes "output":
    Return element-wise multiplication of "output" and (1 - "output")

Define "relu" function that takes "x":
    Return element-wise maximum of 0 and "x"

Define "relu_derivative" function that takes "x":
    Return element-wise conditional array where "x" > 0: 1, else: 0

Define "cross_entropy_derivative" function that takes "y_true" and "y_pred":
    Return element-wise subtraction of "y_pred" and "y_true"

Define class "Layer_Dense":
    Initialize with "n_inputs" and "n_neurons":
        Initialize "weights" as random array of shape (n_neurons, n_inputs)
        Initialize "biases" as zeros array of shape (n_neurons, 1)
    Define "forward" method that takes "inputs":
        Compute "output" as dot product of "weights" and "inputs" plus "biases"

Create instance "hidden_layer" of "Layer_Dense" with n_inputs=784 and n_neurons=32
Create instance "output_layer" of "Layer_Dense" with n_inputs=32 and n_neurons=10

For each "j" in range(EPOCHS):
    Initialize "total_loss" to 0
    For each "i" in range(60000):
        Compute "z_1" as dot product of "hidden_layer.weights" and "X[:,i:i+1]" plus "hidden_layer.biases"
        Compute "a_1" using "relu" activation on "z_1"
        Compute "z_2" as dot product of "output_layer.weights" and "a_1"
        Compute "a_2" using "softmax" activation on "z_2"
        Compute "loss" as negative sum of element-wise multiplication of "y_train[:, i:i+1]" and log of "a_2" plus small epsilon
        Add "loss" to "total_loss"
        Compute "error_term_output" using "cross_entropy_derivative"
        Compute "error_term_hidden" as dot product of transpose of "output_layer.weights" and "error_term_output" element-wise multiplied by derivative of "relu" on "z_1"
        Update "output_layer.weights" using gradient descent update
        Update "hidden_layer.weights" using gradient descent update
        Update "hidden_layer.biases" using gradient descent update
    Compute "mean_loss" as "total_loss" divided by 60000
    Print epoch number, mean loss, and process indicator
Print "Neural network trained successfully!"

"data_test" = Read CSV from file "data/mnist_test.csv"
"x_test" = Select all rows from "data_test" except the first column and normalize by dividing by 255
"y_test" = Select only the first column from "data_test" and squeeze dimensions

Define function "prediction" that takes "input_", "y_true", and "n_cases":
    Initialize "hits" to 0
    For each "i" in range "n_cases":
        Compute "z1" as dot product of "hidden_layer.weights" and "input_[:,i:i+1]" plus "hidden_layer.biases"
        Compute "a1" using "relu" activation on "z1"
        Compute "z2" as dot product of "output_layer.weights" and "a1"
        Compute "a2" using "softmax" activation on "z2" and transpose
        Compute "pred" as index of maximum element in "a2"
        Print prediction and real values and whether it's correct or incorrect
        If "pred" equals "y_true[i]":
            Increment "hits"
    Return "Precision" as string representation of "hits" divided by "n_cases"

Call "prediction" function with "x_test", "y_test", and 100 as arguments
