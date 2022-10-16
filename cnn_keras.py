import os
import tensorflow as tf
import cv2 #pip install opencv-python - si es necesario cambiar el interprete (ctrl + shift + p))      
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Para los dataset consultar en Tensorflow datasets (https://www.tensorflow.org/datasets/catalog/overview)
#El data set de humans or horses se encuentra aqui https://laurencemoroney.com/datasets.html

#Dirección de imagenes para entrenamiento y validación
entrenamiento = 'C:/Users/nfons/Documents/Universidad/Proyecto de grado/Redes/Dataset/Entrenamiento'
validacion = 'C:/Users/nfons/Documents/Universidad/Proyecto de grado/Redes/Dataset/Validacion'

ListaTrain = os.listdir(entrenamiento)
ListaTest = os.listdir(validacion)

#Se establecen parametros
ancho, alto = 200, 200

#listas para entrenamiento
etiquetas = []
fotos = []
datos_train = []
con = 0

#Listas para validación
etiquetas2 = []
fotos2 = []
datos_vali=[]
con2 = 0

#Se extrae en una lista las fotos y etiquetas
#Entrenamiento
for nameDir in ListaTrain:
    nombre = entrenamiento + '/' + nameDir #se leen las fotos

    for fileName in os.listdir(nombre): #se asignan etiquetas a cada foto
        etiquetas.append(con) #valores de etiqueta, 0 a la primera y 1 a la segunda
        img = cv2.imread(nombre + '/' + fileName, 0) #se leen las fotos
        img = cv2.resize(img, (ancho, alto),    interpolation=cv2.INTER_CUBIC) #se redimensionan las fotos
        img = img.reshape(ancho, alto, 1) #se convierten en matrices
        datos_train.append([img, con]) #se agregan a la lista de datos
        fotos.append(img) #se agregan a la lista de fotos
    
    con += 1

#Validación
for nameDir2 in ListaTest:
    nombre2 = validacion + '/' + nameDir #se leen las fotos

    for fileName2 in os.listdir(nombre2): #se asignan etiquetas a cada foto
        etiquetas2.append(con2) #valores de etiqueta, 0 a la primera y 1 a la segunda
        img2 = cv2.imread(nombre2 + '/' + fileName2, 0) #se leen las fotos
        img2 = cv2.resize(img2, (ancho, alto),    interpolation=cv2.INTER_CUBIC) #se redimensionan las fotos
        img2 = img2.reshape(ancho, alto, 1) #se convierten en matrices
        datos_vali.append([img2, con2]) #se agregan a la lista de datos
        fotos2.append(img2) #se agregan a la lista de fotos
    
    con2 += 1
         
#Normalizacion de imagenes
fotos = np.array(fotos).astype(float)/255
print (fotos.shape)
fotos2 = np.array(fotos2).astype(float)/255
print (fotos2.shape)
#Se convierten las etiquetas en un array
etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)