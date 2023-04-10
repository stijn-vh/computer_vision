# Integrate and adapt AlexNet to use 172x172x1 inputs
from keras import Input
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from keras.losses import SparseCategoricalCrossentropy
from single_stream_model import conv_block

conv1_params = {
    'filters': 96,
    'kernel_size': (7, 7),
    'padding': 'same',
    'strides': (2, 2),
    'batch_norm': True,
    'pooling_size': (2, 2),
    'pooling_strides': (1, 1)
}

conv34_params = {
    'filters': 512,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'batch_norm': False
}

fc_params = {
    'units': 4096,
    'dropout': 0.9
}

def conv_block(x, params):
    x = Conv2D(
        filters = params['filters'],
        kernel_size = params['kernel_size'],
        padding = params['padding'],
        strides = params['strides'],
    )(x)

    if params['batch_norm']:
        x = BatchNormalization()(x)

    if params['pooling_size'] and params['pooling_size']:
        x = MaxPooling2D(
            pool_size = params['pooling_size'],
            strides = params['pooling_strides']
        )

    return Activation(params['act_func'])(x)

def fully_connected(x, params):
    x = Dense(x['units'])

    if params['dropout_rate'] > 0:
        x = Dropout(params['dropout_rate'])(x)
        
    return Activation(params['act_func'])(x)

def create_first_four_conv_blocks(input_shape):
    inputs = Input(shape = input_shape)

    conv1 = conv_block(inputs, conv1_params)

    conv2_params = conv1_params
    conv2_params['filters'] = 256
    conv2_params['kernel_size'] = (5, 5)

    conv2 = conv_block(conv1, conv2_params)
    conv3 = conv_block(conv2, conv34_params)
    conv4 = conv_block(conv3, conv34_params)

    return conv4, inputs

def create_dense_layers(x):
    flatten = Flatten(x)

    fc1 = fully_connected(flatten, fc_params)

    fc_params['units'] = 2048
    fc2 = fully_connected(fc1, fc_params)

    return Activation('softmax')(fc2)


def create_stream_model(input_shape):
    conv4, inputs = create_first_four_conv_blocks(input_shape)

    conv5_params = conv34_params
    conv5_params['pooling_size'] = (2, 2)
    conv5_params['pooling_strides'] = (1, 1)

    conv5 = conv_block(conv4, conv5_params)

    outputs = create_dense_layers(conv5)

    model = Model(inputs, outputs)
    model.compile(optimizer = 'adam',
                  loss = SparseCategoricalCrossentropy(),
                  metrics = ['accuracy'])
    
    return model





