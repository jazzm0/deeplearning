import keras
from keras.datasets import fashion_mnist
from deeplearning2020 import Submission

# 0 	T-shirt/top
# 1 	Trouser
# 2 	Pullover
# 3 	Dress
# 4 	Coat
# 5 	Sandal
# 6 	Shirt
# 7 	Sneaker
# 8 	Bag
# 9 	Ankle boot

(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()

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

model.fit(train_images, train_label_vectorized, epochs=30)

eval_loss, eval_accuracy = model.evaluate(test_images, test_label_vectorized)
print(eval_loss, eval_accuracy)

Submission('a70a2614a4468a25eb66a1113c846e31', '2', model).submit()
