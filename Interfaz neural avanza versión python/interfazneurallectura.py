from __future__ import unicode_literals, print_function,absolute_import,division
import tensorflow as tf
import tensorflow as tfds


import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()

logger.setLevel(logging.ERROR)


dataset,metadata = tfds.load('minst', as_supervised= True, with_info=True)
train_dataset,test_dataset = dataset['train'], dataset['test']

class_names =[
    'cero','uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 
    'siete', 'ocho','nueve'
]

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

#Normalizar: Numeros de 0 a 255, que sean de 0 a 1
def noemalize(images,labels):
    images = tf.cast(images,tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map("normalize")    
test_dataset = train_dataset.map("normalize")

#Estructura de la red
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(64,activation=tf.nn.relu),
    tf.keras.layers.Dense(64,activation=tf.nn.relu),
    tf.keras.layers.Dense(64,activation=tf.nn.softmax)#para clasificación
])

#Indicar las funciones a utilizaar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Aprendizaje por lotes de 32 cada lote
BATCHIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHIZE)
test_dataset = test_dataset.batch(BATCHIZE)

#Realizar el apredizaje
model.fit(
    train_dataset,epochs=5,
    steps_per_epoch=math.ceil(num_train_examples/BATCHIZE)#No sera necesario pronto
)

#Evaluar nuestro modelo ya entrenado contra el dataset de pruebas
test_loss,test_accurancy = model.evaluate(
    test_dataset,steps=math.ceil(num_test_examples/32)
)

print("Resultado en las pruebas:",test_accurancy)

for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_images = test_labels.numpy()
    predictions = model.predict(test_images)
