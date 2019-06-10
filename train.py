import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense,Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications


#Creacion de la red neuronal, red convulsional
def mymodelo():
    cnn = Sequential()  #varias capas apiladas sobre ella
    #Agregamos primera capa Convolucional con su respectivo pool
    cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=tamano_pool))
    #Segunda Capa
    cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
    cnn.add(MaxPooling2D(pool_size=tamano_pool))
    #Tercera Capa
    cnn.add(Convolution2D(filtrosConv3, tamano_filtro3, padding ="same"))
    cnn.add(MaxPooling2D(pool_size=tamano_pool))

    cnn.add(Flatten())  #hacemos la imagen plana en una sola dimension
    cnn.add(Dense(256, activation='relu')) #cantidad de neuronas 256
    cnn.add(Dropout(0.5)) #aqui le dicemos que durante el entrenamiento le apagamos el 50% de las neuronas a cada paso, se hace para evitar sobreajustar
    cnn.add(Dense(clases, activation='softmax')) #ultima capa, capa de output
    return cnn    

#modelo de transfer learning 

def tlmodel():
 vgg=applications.vgg16.VGG16()
 cnn1 =Sequential()
 for cap in vgg.layers:
     cnn1.add(cap)
 cnn1.layers.pop()
 for layer in cnn1.layers:
     layer.trainable=False
 cnn1.add(Dense(clases,activation='softmax'))
 return cnn1        

K.clear_session()

training_data='./dataSet-simple/train'
validation_data='./dataSet-simple/test'

#Parametros

epocas=20
longitud, altura=100,100
batch_size = 32 #numero de imagenes a procesar en cada uno de los pasos
pasos = 1000  #es el numero de veces que se procesara la informacion en cada una de las epocas
validation_steps = 300 #es el numero de veces que probaremos el algoritmo con los datos de validacion
filtrosConv1 = 16  #filtros a aplicar durante la primera convulsion profundidad 32
filtrosConv2 = 32   #filtros a aplicar durante la segunda convulsion profundidad 64
filtrosConv3= 64
tamano_filtro1 = (4,4)
tamano_filtro2 = (3,3)
tamano_filtro3= (2,2)
tamano_pool = (2,2)
clases = 7
lr = 0.0004 #learning rate tamaño del os ajustes que hace nuestra cnn para llegar a la respuesta correcta


##Preparamos nuestras imagenes


training_datagen = ImageDataGenerator(
    rescale=1. /255,   #rescalar las imagenes a valores de pixeles de 0 a 1
    shear_range=0.2,    #generar nuestras imagenes pero las inclina 
    zoom_range=0.2,     #les hace zoom a algunas imagenes    
    horizontal_flip=True)  #invierte una imagen 

test_datagen = ImageDataGenerator(rescale=1. /255)

entrenamiento_generador = training_datagen.flow_from_directory(
    training_data,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validation_datagen = test_datagen.flow_from_directory(
    validation_data,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical') #este es el tipo de clasificacion

cnn1=mymodelo()
cnn1.summary()
cnn1.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])    



#entrenamiento y save del modelo
cnn1.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validation_datagen,
    validation_steps=validation_steps
   )

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn1.save('./modelo/modelo.h5') #estructura del modelo
cnn1.save_weights('./modelo/pesos.h5')   #peso de cada una de las capas          