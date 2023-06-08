# we are going to use the classic MNIST dataset to perform classifications on handwritten digits 0–9

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# load the MNIST dataset straight from TensorFlow by calling mnist.load_data(). It will even have the training and testing datasets separated already. A total of 70,000 handwritten digit images are in this dataset, each with dimensions of 28 by 28 pixels.
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print the first training record, a "5" image
print(X_train[0])

# rescale data each px b/w 0 to 1
# This will make the training more efficient as there is less numeric range to travel through.
X_train = X_train / 255
X_test = X_test / 255

# reshape each 2-dimensional matrix of pixels into a 1-dimensional vector, stacking each row of pixels into a single row.
print(X_train.shape)
n_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], n_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], n_pixels)).astype('float32')

print(y_train[0:5])

# The Y_train and Y_test will contain the digits 0–9 labeling each image, and the first 5 records have labels 5 0 4 1 9.
# To make this compatible with our neural network, we need to perform one-hot encoding which means each label becomes a vector of 1s and 0s, where the position of the 1 indicates which label it is.
# We can perform this transformation using to_categorical().

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[0:5])

# The first layer we add will be a dense hidden layer with a relu activation function.
# It will expect the number of pixels (784) as an input and output 784 values, applying the weights, biases, and activation ReLU function.
# the putput layer is multi layered perceptron

model = Sequential()
model.add(Dense(n_pixels, input_shape=(n_pixels,), activation='relu'))

# We want the number of output nodes to be the number of classes, which would be 10 since there are digits 0–9.
# We will also use the softmax function, which will rescale all the outputted probabilities for each category so they add up to 1.0.
# The digit output node with the highest probability will be the prediction.

num_classes = y_test.shape[1]
model.add(Dense(num_classes, activation='softmax'))

# compile the model, fit the data, and evaluate the test dataset.
# We will use the MeanSquaredError as our loss function.
# We will do 10 passes/epochs through the data, updating the parameters in batches of 100.

model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Score: {scores[1]}")