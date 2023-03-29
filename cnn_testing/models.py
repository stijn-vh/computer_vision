#Five models with slightly different architectures
from keras import Input
from keras import Model
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Rescaling, RandomZoom, RandomFlip, RandomRotation
from keras import Sequential
from keras.metrics.accuracy_metrics import TopKCategoricalAccuracy
import math

def conv_block(x, conv1_filters, conv1_kernel_size, activation_function, batch_norm, padding):
    x = Conv2D(conv1_filters,
               conv1_kernel_size,
               padding = padding)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    out = Activation(activation_function)(x)
    return out

def dense_block(x, units, activation_function, dropout_rate, batch_norm):
    x = Dense(units)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    out = Activation(activation_function)(x)
    return out

def cnn_model(input_shape=(28, 28, 1), num_classes=10,
              conv1_filters=6, conv1_kernel_size=(5, 5),
              pool_size=(2, 2),
              pool_stride=(2, 2),
              conv2_filters=16, conv2_kernel_size=(5, 5),
              dense1_units=120,
              dense2_units=84,
              activation_function="tanh",
              extra_conv=False,
              batch_norm=False,
              dropout_rate=0,
              data_augmentation=False):
    # The default parameters define the lenet-5 model.

    inputs = Input(shape=input_shape)
    if data_augmentation:
        trainAug = Sequential([
            Rescaling(scale=1.0 / 255),
            RandomFlip("horizontal_and_vertical"),
            RandomZoom(
                height_factor=(-0.05, -0.15),
                width_factor=(-0.05, -0.15)),
            RandomRotation(0.3)
        ])
        x = trainAug(inputs)
    else:
        x = inputs

    x = conv_block(x, conv1_filters, conv1_kernel_size, activation_function, batch_norm, padding='same')
    x = AveragePooling2D(pool_size=pool_size, strides=pool_stride)(x)
    x = conv_block(x, conv2_filters, conv2_kernel_size, activation_function, batch_norm, padding='valid')
    x = AveragePooling2D(pool_size=pool_size, strides=pool_stride)(x)
    if extra_conv:
        x = conv_block(x, 32, (3, 3), activation_function, batch_norm, padding='valid')

    x = Flatten()(x)
    x = dense_block(x, dense1_units, activation_function, dropout_rate, batch_norm)

    x = dense_block(x, dense2_units, activation_function, dropout_rate, batch_norm)

    outputs = Dense(num_classes)(x)

    model = Model(inputs=inputs,
                  outputs=outputs)

    
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

