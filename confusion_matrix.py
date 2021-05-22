
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

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

# SHOW CONFUSION MATRIX

list_file_potholes = os.listdir(path_potholes)
list_file_normal = os.listdir(path_normal)

test_labels = []
test_preds = []

# se obtienen las imagenes de baches
for index, filename in enumerate(list_file_potholes[:100]):
    path_img = os.path.join(path_potholes, filename)

    test_labels.append('bache')
    
    # se obtiene la imagen
    img_color = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_color, (64, 64)) #64 x 64 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, 64, 64, 3)
    model_out = model.predict([data])
    
    # se establece la leyenda de salida
    if np.argmax(model_out) == 0:
        test_preds.append('bache')
    else:
        test_preds.append('sin_bache')

# se obtienen las imagenes sin baches
for index, filename in enumerate(list_file_normal[:100]):
    path_img = os.path.join(path_normal, filename)

    test_labels.append('sin_bache')

    # se obtiene la imagen
    img_color = cv2.imread(path_img, cv2.IMREAD_COLOR) #RBG Level
    img_pixel = cv2.resize(img_color, (64, 64)) #64 x 64 pixel
    
    # se convierte la imagen a matriz y se realiza la evaluacion
    data = img_pixel.reshape(1, 64, 64, 3)
    model_out = model.predict([data])
    
    # se establece la leyenda de salida
    if np.argmax(model_out) == 0:
        test_preds.append('bache')
    else:
        test_preds.append('sin_bache')

# se genera la matriz de confusion del modelo
cm = confusion_matrix(test_labels, test_preds)

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(14,14), class_names=('bache', 'sin_bache'))
plt.show()
# se guarda el grafico de la matriz generada
fig.savefig('./model/conf_matrix.png')