import numpy as np

def CrossCorrelation(x_matrix, kernel):
    k_shape = kernel.shape[0]
    new_dim = x_matrix.shape[0]-kernel.shape[0]+1 #r-k+1
    output_matrix = np.zeros((new_dim,new_dim))
    for i in range(new_dim):
        for j in range(new_dim):
            output_matrix[i,j] = np.sum( x_matrix[i:i+k_shape,j:j+k_shape] * kernel )
    return output_matrix

matriz_6x6 = np.random.randint(1, 11, size=(6, 6))
kernel_3x3 = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])

print("Imagen:")
print(matriz_6x6)
print("kernel")
print(kernel_3x3)

print(CrossCorrelation(matriz_6x6,kernel=kernel_3x3))
