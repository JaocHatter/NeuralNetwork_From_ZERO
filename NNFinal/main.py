import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import Sequential
from layers import Dense
EPOCH = 800
LEARNING_RATE = 0.000001 
np.random.seed(7) #agregamos una semilla

#Prepare our data

data_set=pd.read_csv("data/mnist_train.csv")
X = data_set.iloc[0:,1:].to_numpy() #Normalize
y_labels = data_set.iloc[:,0:1].to_numpy().squeeze()
data_length = len(y_labels)

X = X/255
image_length = len(X[1])

#Apply One Hot Encodding    

y_train_labels = np.zeros((data_length,10))

for i in range(data_length):
    y_train_labels[i][y_labels[i]]=1
    
    
#Creando el modelo
model = Sequential()

model.add(Dense(32,"relu",784))
model.add(Dense(10,"softmax",32))

model.compile("GD","categorical_crossentropy",False,lr = LEARNING_RATE )

print(len(model.layers))
model.fit(X,y_train_labels,EPOCH)

#History 

plt.plot(model.losses)
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