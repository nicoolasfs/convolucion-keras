import os
import tensorflow as tf
import cv2 #pip install opencv-python - si es necesario cambiar el interprete (ctrl + shift + p))      
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Para los dataset consultar en Tensorflow datasets (https://www.tensorflow.org/datasets/catalog/overview)

#Dirección de imagenes para entrenamiento y validación
entrenamiento = 'C:\Users\nfons\Documents\Universidad\Proyecto de grado\Redes\Dataset\Entrenamiento'
validacion = 'C:\Users\nfons\Documents\Universidad\Proyecto de grado\Redes\Dataset\Validacion'

ListaTrain = os.listdir(entrenamiento)
ListaTest = os.listdir(validacion)

#Se establecen parametros
ancho, alto = 200, 200

#listas para entrenamiento
etiquetas = []
fotos = []
datos_train = []
con = 


