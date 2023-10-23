import numpy as np
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.rand(n_inputs,n_neurons) #W
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.biases # XW + b
