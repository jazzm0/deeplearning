import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(train_images, train_label), (test_images, test_label) = data.load_data()

print(train_images.shape)
print(test_images.shape)
print(np.min(train_images))
print(np.max(train_images))

print(f"Label {train_label[0]}")
total_classes = 10
train_label_vectorized = keras.utils.to_categorical(train_label, total_classes)
print(f"Label after vectorize {train_label_vectorized[0]}")

plt.imshow(train_images[0])
plt.colorbar()
plt.show()
normalized_train_images = train_images / 255
plt.imshow(normalized_train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()
print()
