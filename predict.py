import tensorflow as tf
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model
import os
import glob

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn1 = load_model(modelo)
cnn1.load_weights(pesos_modelo)
test_images="./test_examples/*.jpg"

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn1.predict(x)
 
  result = array[0]
  answer = np.argmax(result)
  print(file)
  print(answer)
  if answer == 0:
    print("pred: Manzana")
  elif answer == 1:
    print("pred: Banana")
  elif answer == 2:
    print("pred: Banana Lady Finger")
  elif answer == 3:
    print("pred: Kiwi")
  elif answer == 4:
    print("pred: Lemon")  
  elif answer == 5:
    print("pred: Lime")
  elif answer == 6:
    print("pred: Mango")
  elif answer == 7:
    print("pred: Melon")     
  elif answer == 8:
    print("pred: Orange")
  elif answer == 9:
    print("pred: Tomate")                           
  return answer

list_img=glob.glob(test_images)
print(list_img)

for imgs in list_img:
  predict(imgs)