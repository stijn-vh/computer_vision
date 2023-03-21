#Five models with slightly different architectures
from keras import Input
from keras import Model
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from keras.losses import SparseCategoricalCrossentropy

def conv_block(x, conv1_filters, conv1_kernel_size, activation_function, batch_norm, pool_size, pool_stride):
    x = Conv2D(conv1_filters,
               conv1_kernel_size,
               padding = 'same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    out = Activation(activation_function)(x)
    return out

def cnn_model(input_shape=(28, 28, 1), num_classes=10,
            conv1_filters=6, conv1_kernel_size=(5, 5),
            pool_size=(2, 2),
            pool_stride= (2,2),
            conv2_filters=16, conv2_kernel_size=(5, 5),
            dense1_units=120,
            dense2_units=84,
            activation_function= "tanh",
            batch_norm = False,
            dropout_rate = 0):
    #The default parameters define the lenet-5 model.

    inputs = Input(shape=input_shape)

    x = conv_block(inputs, conv1_filters, conv1_kernel_size, activation_function, batch_norm, pool_size, pool_stride)
    x = AveragePooling2D(pool_size=pool_size, strides=pool_stride)(x)
    x = conv_block(x, conv2_filters, conv2_kernel_size, activation_function, batch_norm, pool_size, pool_stride)
    x = AveragePooling2D(pool_size=pool_size, strides=pool_stride)(x)

    x = Flatten()(x)
    x = Dense(dense1_units,
              activation=activation_function)(x)
    if dropout_rate>0:
        x = Dropout(dropout_rate)(x)
    x = Dense(dense2_units,
              activation=activation_function)(x)
    # if dropout_rate>0:
    #     x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes)(x)

    model = Model(inputs=inputs,
                  outputs=outputs)

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

