#Five models with slightly different architectures

from keras.layers import Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Concatenate, Input, BatchNormalization, Activation

#Input -> Convolutional -> BatchNormalization -> Pooling -> Dropout -> Convolutional
# -> ReLU Activation -> Pooling -> Flatten -> Dense -> Dense (Output)

def model(input_shape= (28,28,1), n_classes = 10, batch_size =32,
          action_function = "tanh", batch_norm = False, kernel_size = (5,5), num_filters = []):

    inputs = Input(shape = input_shape, batch_size = batch_size)
    x = Conv2D( # Number of filters
                  kernel_size,
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(action_function)(x)


    model = model(inputs= inputs, output = x)
    return model

def model_dropout():
