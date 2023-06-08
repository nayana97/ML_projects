import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv("https://tinyurl.com/y2qmhfsr")

# Extract the input columns, scale down by 255
X = (data.iloc[:, 0:3].values / 255.0)
Y = data.iloc[:, -1].values

# Split train and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

# declare the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(3, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

loss_fn = tf.keras.losses.MeanSquaredError()

# compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Dataset Score: {scores[1]}")
