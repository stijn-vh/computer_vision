import numpy as np
from keras.layers import RandomZoom, RandomFlip, RandomRotation, RandomCrop, RandomBrightness, RandomContrast
from keras import Sequential
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras_preprocessing.image import ImageDataGenerator
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


def augment_data(X_train, Y_train):
    X_train =np.expand_dims(X_train, axis =-1)

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = train_datagen.flow(
        X_train,
        Y_train,
    )
    return train_gen


def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.random_crop(image, size = (28, 28,1))
    return image, label



