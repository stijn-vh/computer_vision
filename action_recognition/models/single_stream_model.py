# Integrate and adapt AlexNet to use 172x172x1 inputs
from keras import Input
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf

conv1_params = {
    'filters': 48,
    'kernel_size': (7, 7),
    'padding': 'same',
    'strides': (2, 2),
    'batch_norm': True,
    'pooling_size': (3, 3),
    'pooling_strides': (2, 2),
    'act_func': "relu"
}

conv2_params = conv1_params.copy()
conv2_params['filters'] = 64
conv2_params['kernel_size'] = (5, 5)

conv34_params = {
    'filters': 96,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'batch_norm': False,
    'act_func': "relu"
}

conv5_params = conv34_params.copy()
conv5_params['pooling_size'] = (3, 3)
conv5_params['pooling_strides'] = (2, 2)

fc1_params = {
    'units': 256,
    'dropout': 0.5,
    'act_func': "relu"
}

fc2_params = fc1_params.copy()
fc2_params['units'] = 128


def conv_block(x, params):
    x = Conv2D(
        filters=params['filters'],
        kernel_size=params['kernel_size'],
        padding=params['padding'],
        strides=params['strides'],
    )(x)

    if params['batch_norm']:
        x = BatchNormalization()(x)

    x = Activation(params['act_func'])(x)
    if 'pooling_size' in params and 'pooling_strides' in params:
        x = MaxPooling2D(
            pool_size=params['pooling_size'],
            strides=params['pooling_strides']
        )(x)
    return x


def fully_connected(x, params):
    x = Dense(params['units'])(x)

    if 'dropout_rate' in params:
        x = Dropout(params['dropout_rate'])(x)

    return Activation(params['act_func'])(x)


def create_single_stream_model(input_shape, num_classes=12):
    inputs = Input(shape=input_shape)

    conv1 = conv_block(inputs, conv1_params)

    conv2 = conv_block(conv1, conv2_params)
    conv3 = conv_block(conv2, conv34_params)
    conv4 = conv_block(conv3, conv34_params)

    conv5 = conv_block(conv4, conv5_params)

    flatten = Flatten()(conv5)

    fc1 = fully_connected(flatten, fc1_params)

    fc2 = fully_connected(fc1, fc2_params)

    dense_output = Dense(num_classes)(fc2)

    outputs = Activation('softmax')(dense_output)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])
    return model
