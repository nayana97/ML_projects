import numpy as np
import pandas as pd

# dataset: https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/light_dark_font_training_set.csv
all_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")

# Extract the input columns, scale down by 255
X = (all_data.iloc[:, 0:3].values / 255.0)
Y = all_data.iloc[:, -1].values

# print(X, "->", Y)
# Build neural network with weights and biases
# with random initialization

w_hidden = np.random.rand(3, 3)
w_output = np.random.rand(1, 3)

b_hidden = np.random.rand(3, 1)
b_output = np.random.rand(1, 1)

# Activation functions
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1/(1+np.exp(-x))

# Z1 = W(hidden) X + B(hidden)
def farword_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

# Calculate accuracy
test_predictions = farword_prop(X.transpose())[3] # grab only A2
test_comparions = np.equal((test_predictions >= .5).flatten().astype(int), Y)
accuracy = sum(test_comparions.astype(int) / X.shape[0])
print("ACCURACY: ", accuracy)