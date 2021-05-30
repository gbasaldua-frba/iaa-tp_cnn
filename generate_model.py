import cv2
import numpy as np
import random
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# vector de imagenes
images = []
# paths de imagenes
path_potholes = './data/train/potholes'
path_normal = './data/train/normal'
list_file_potholes = os.listdir(path_potholes)
list_file_normal = os.listdir(path_normal)
target_dir = './model/'

# se agregan las imagenes de los baches
for filename in tqdm(list_file_potholes):
    path_img = os.path.join(path_potholes, filename)
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    # print(path_img)
    img_pixel = cv2.resize(img_pixel, (64, 64)) #64 x 64 pixel
    images.append([np.array(img_pixel), np.array([1, 0])])

# se agregan las imagenes de las carreteras sin baches
for filename in tqdm(list_file_normal):
    path_img = os.path.join(path_normal, filename)
    img_pixel = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    # print(path_img)
    img_pixel = cv2.resize(img_pixel, (64, 64)) #64 x 64 pixel
    images.append([np.array(img_pixel), np.array([0, 1])])

random.shuffle(images)
train = images

# se crean la matrices de datos y labels
train_data = np.array([i[0] for i in train]).reshape(-1, 64, 64, 3)
train_label = np.array([i[1] for i in train])


# CREATE MODEL

print("Comienzo del proceso de armado del modelo")

# creacion de modelo secuencial
model = Sequential()

# capa input de 64x64 pixeles y 3 dimensiones RGB
model.add(InputLayer(input_shape=[64, 64, 3]))

# primera capa conv+relu+pool
model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

# segunda capa conv+relu+pool
model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

# tercera capa conv+relu+pool
model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

# dropout y capa plana
model.add(Dropout(0.4))
model.add(Flatten())

# capa de activacion y salida softmax
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(2, activation='softmax'))

# funcion para la optimizacion
optimizer = Adam(learning_rate=1e-4)

# compilacion del modelo
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("Modelo compilado")

# se muestra la arquitectura del modelo generado
model.summary()

# entrenamiento del modelo
history = model.fit(x=train_data, y=train_label, epochs=200, batch_size=128, validation_split=0.1)

print("Fin de entrenamiento del modelo")

# impresion de las metricas del modelo craado
scores = model.evaluate(train_data, train_label, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# MODEL TO JSON

# se crea la carpeta el almacenamiento del modelo
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

# creacion y alamacenamiento del modelo en formato jason
model_json = model.to_json()
with open("./model/model.json", "w") as json_file:
    json_file.write(model_json)

# almacenamiento de los pesos a formato h5
model.save_weights("./model/weights.h5")
print("Se ha guardado el modelo generado")


# SHOW MODEL TRAIN SEQUENCE

# se muestra el avance del entrenamiendo en cuanto a la presicion
plt.figure()
plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('accurancy')
plt.legend()
plt.savefig('./model/accur.png')
plt.close()

# se muestra el avance del entrenamiento en cuanto a la perdida
plt.figure()
plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./model/loss.png')
plt.close()
