import numpy as np
from keras import Sequential

import tensorflow as tf
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

def grayscale_images(images):
    return images.astype('float32') / 255



def load_mnist_data():
    # Load the Fashion MNIST dataset from Keras
    (X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()

    # Split the full training set into training and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, test_size=0.3, random_state=1)

    # Normalize the pixel values to be between 0 and 1
    X_train = grayscale_images(X_train)
    X_valid = grayscale_images(X_valid)
    X_test = grayscale_images(X_test)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

"""
def to_tf_data(X_train, Y_train, data_augmentation = False):

    X_train = np.expand_dims(X_train, axis=-1)
    Y_train = np.expand_dims(Y_train, axis=-1)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    if data_augmentation:
        train_ds = train_ds.map(lambda x, y: (trainAug(x), y),
		 num_parallel_calls=tf.data.AUTOTUNE)
        print("debug")
    return train_ds
def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.random_crop(image, size = (28, 28,1))
    return image, label
"""


