import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(train_images, train_label), (test_images, test_label) = data.load_data()

train_images = train_images / 255
test_images = test_images / 255

total_classes = 10

train_label_vectorized = keras.utils.to_categorical(train_label, total_classes)
test_label_vectorized = keras.utils.to_categorical(test_label, total_classes)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input
    keras.layers.Dense(128, activation='sigmoid'),  # hidden
    keras.layers.Dense(total_classes, activation='sigmoid')  # output
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.fit(train_images, train_label_vectorized, epochs=20)

eval_loss, eval_accuracy = model.evaluate(test_images, test_label_vectorized)
print(eval_loss, eval_accuracy)
