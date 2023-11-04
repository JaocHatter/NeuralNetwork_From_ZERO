import cv2
import numpy as np

def CrossCorrelation(x_matrix, kernel):
    k_s = kernel.shape[0]
    new_dim = np.array(x_matrix.shape) - np.array(kernel.shape) + 1 #r-k+1
    output_matrix = np.zeros(tuple(new_dim))
    for i in range(x_matrix.shape[0] - k_s + 1):
        for j in range(x_matrix.shape[1] - k_s + 1):
            output_matrix[i,j] = np.sum( x_matrix[i:i+k_s,j:j+k_s] * kernel )
    return output_matrix

arreglo = cv2.imread('/home/jaoc/Escritorio/NeuralNetwork_From_Zero/imagenes/perro.jpeg')

print(arreglo.shape)

canal_azul = arreglo[:, :, 0]
canal_rojo = arreglo[:, :, 2]

canal_verde = arreglo[:, :, 1]
kernel_negativo = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])

canal_azul_result = CrossCorrelation(x_matrix=canal_azul,kernel=kernel_negativo)
canal_verde_result = CrossCorrelation(canal_verde, kernel_negativo)
canal_rojo_result = CrossCorrelation(canal_rojo, kernel_negativo)

# Asegurarse de que los valores estén en el rango adecuado (0-255)
canal_azul_result = np.clip(canal_azul_result, 0, 255).astype(np.uint8)
canal_verde_result = np.clip(canal_verde_result, 0, 255).astype(np.uint8)
canal_rojo_result = np.clip(canal_rojo_result, 0, 255).astype(np.uint8)

# Unir los canales para formar la imagen resultante
imagen_resultante = cv2.merge((canal_azul_result, canal_verde_result, canal_rojo_result))

# Mostrar la imagen resultante
cv2.namedWindow('Imagen', cv2.WINDOW_NORMAL) 
cv2.imshow('Imagen', imagen_resultante)
cv2.resizeWindow('Imagen',800,600)
cv2.waitKey(0)
cv2.destroyAllWindows()





