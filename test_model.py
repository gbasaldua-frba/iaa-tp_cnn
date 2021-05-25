import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras as kr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# GET MODEL

# path de imagenes
path_potholes = './data/test/potholes'
path_normal = './data/test/normal'

# se obtiene el modelo generado
json_file = open('./model/model.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)

# se obtienen los pesos
model.load_weights("./model/weights.h5")
print("Modelo cargado")

# se compila nuevamente el modelo
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# SHOW RESULT
list_file_potholes = os.listdir(path_potholes)
list_file_normal = os.listdir(path_normal)

fig = plt.figure(figsize=(14, 14))

# se obtienen cinco imagenes de baches
for index, filename in enumerate(list_file_potholes[:10]):
    path_img = os.path.join(path_potholes, filename)
    y = fig.add_subplot(6, 5, index+1)

    # se obtiene la imagen
    img_color = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_color, (64, 64)) #64 x 64 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, 64, 64, 3)
    model_out = model.predict([data])
    
    # se establece la leyenda de salida
    if np.argmax(model_out) == 0:
        str_label = 'pred: Bache'
    else:
        str_label = 'pred: Sin Bache'

    # se muestra la imagen junto con su leyenda
    y.imshow(img_color)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

# se obtienen cinco imagenes de carreteras sin baches
for index, filename in enumerate(list_file_normal[:10]):
    path_img = os.path.join(path_normal, filename)
    y = fig.add_subplot(6, 5, index+6)

    # se obtiene la imagen
    img_color = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_color, (64, 64)) #64 x 64 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, 64, 64, 3)
    model_out = model.predict([data])
    
    # se establece la leyenda de salida
    if np.argmax(model_out) == 0:
        str_label = 'pred: Bache'
    else:
        str_label = 'pred: Sin Bache'

    y.imshow(img_color)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)


plt.show()
