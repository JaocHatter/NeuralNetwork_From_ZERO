import numpy as np
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
