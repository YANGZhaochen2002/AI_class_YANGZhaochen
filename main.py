import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Load the handwritten digits dataset
digits = datasets.load_digits()

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)


# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Reshape the input data
X_train = X_train.reshape(-1, 8, 8, 1)
X_test = X_test.reshape(-1, 8, 8, 1)

# Convert target labels to one-hot vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Create the CNN model
model = tf.keras.Sequential()

# Add the first convolutional layer






