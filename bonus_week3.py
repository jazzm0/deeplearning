# !pip install deeplearning2020
import tensorflow as tf
import tensorflow_datasets as tfds
from deeplearning2020 import helpers
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow import keras
from deeplearning2020.datasets import ImageWoof
from deeplearning2020 import Submission


# def preprocess(image, label):
#     resized_image = tf.image.resize(image / 255, [300, 300])
#     return resized_image, label
#
#
# train_data = tfds.load('imagenette/320px', split=tfds.Split.TRAIN, as_supervised=True)
# test_data = tfds.load('imagenette/320px', split=tfds.Split.VALIDATION, as_supervised=True)


def preprocess(image, label):
    resized_image = tf.image.resize(image, [300, 300])
    return resized_image, label


train_data, test_data, classes = ImageWoof.load_data()

n_classes = 10
batch_size = 32

train_data = train_data.shuffle(1000)

train_data = train_data.map(preprocess).batch(batch_size).prefetch(1)
test_data = test_data.map(preprocess).batch(batch_size).prefetch(1)

learning_rate = 0.001
momentum = 0.9
dense_neurons = 1000
n_filters = 512

activation = 'elu'

# Inputgröße muss 300x300 Pixel mit 3 RGB Farben betragen
input_layer = Input(shape=(300, 300, 3))

# Convolutional Neural Network
# 6 Convolutional Layers mit jeweils einer Max Pooling Layer
model = Conv2D(filters=256, kernel_size=(7, 7), activation=activation)(input_layer)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(filters=256, kernel_size=(3, 3), activation=activation)(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(filters=n_filters, kernel_size=(3, 3), activation=activation)(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(filters=n_filters, kernel_size=(3, 3), activation=activation)(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(filters=n_filters, kernel_size=(3, 3), activation=activation)(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(filters=n_filters, kernel_size=(3, 3), activation=activation, padding='same')(model)
model = MaxPooling2D((2, 2))(model)

# Fully-Connected-Classifier
model = Flatten()(model)
model = Dense(dense_neurons, activation=activation)(model)
model = Dense(dense_neurons / 2, activation='tanh')(model)

# Output Layer
output = Dense(n_classes, activation="softmax")(model)

CNN_model = Model(input_layer, output)

# Kompilieren des Modells
optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
CNN_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
CNN_model.summary()

history2 = CNN_model.fit(train_data, epochs=12, validation_data=test_data)

helpers.plot_history('Accuracy zweites CNN', history2, 0)
Submission('c1dc649d060fb05ca3486ec58b50fec2', '3', CNN_model).submit()