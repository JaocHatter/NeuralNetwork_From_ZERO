import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1 , 0).astype(float)

def cross_entropy_derivative(y_true,y_pred):
    return y_pred-y_true # -log(y_true/y_pred)    

def softmax(x):
    temp = x - np.max(x , axis=1 , keepdims = True)
    exp_x = np.exp(temp) #cada elemento de la matriz es exponente de e
    suma =  np.sum(exp_x, axis = 1, keepdims=True)
    return exp_x / suma

def mse(y_true,y_pred):
    length_data = y_true.shape[1]
    return (1/length_data) * np.square(y_true-y_pred)

def categorical_crossentropy(y_true,y_pred):
    assert y_true.shape == y_pred.shape, "Las dimensiones de y_true y y_pred deben coincidir."
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred))
    return loss

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

loss_function = {"categorical_crossentropy":categorical_crossentropy,"mean_squared_error":mse}