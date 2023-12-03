import numpy as np
from PIL import Image
import imageio

def preprocess_input(x, v2=True):
    x = x.astype('float32') / 255.0
    if v2:
        x = 2.0 * (x - 0.5)
    return x

def _imread(image_name):
    # Utiliza Image.open de Pillow para cargar la imagen
    image = Image.open(image_name)
    # Convierte la imagen a un arreglo NumPy
    return np.array(image)

def _imresize(image_array, size):
    # Utiliza Image.fromarray de Pillow para crear una imagen desde el arreglo
    image = Image.fromarray(image_array)
    # Cambia el tamaño de la imagen y convierte la imagen redimensionada a un arreglo NumPy
    return np.array(image.resize(size))

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical