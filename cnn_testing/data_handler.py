import numpy as np
from tensorflow import keras
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

def grayscale_images(images):
    return images.astype('float32') / 255

def load_mnist_data():
    # Load the Fashion MNIST dataset from Keras
    (X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()

    # Split the full training set into training and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, test_size=0.5, random_state=25)

    # Normalize the pixel values to be between 0 and 1
    X_train = grayscale_images(X_train)
    X_valid = grayscale_images(X_valid)
    X_test = grayscale_images(X_test)

    Y_train = grayscale_images(Y_train)
    Y_valid = grayscale_images(Y_valid)
    Y_test = grayscale_images(Y_test)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


