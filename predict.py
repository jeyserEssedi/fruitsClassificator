import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

longitud, altura = 224, 224
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn1 = load_model(modelo)
cnn1.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn1.predict(x)
  print(array)
  result = array[0]
  answer = np.argmax(result)
  print(answer)
  if answer == 0:
    print("pred: Banana")
  elif answer == 1:
    print("pred: Kiwi")
  elif answer == 2:
    print("pred: Lemon")
  elif answer == 3:
    print("pred: Lime")
  elif answer == 4:
    print("pred: Mango")  
  elif answer == 5:
    print("pred: Melon")
  elif answer == 6:
    print("pred: Orange")                 
  return answer

predict("Banana_002.jpg")  