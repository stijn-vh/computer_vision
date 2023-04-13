import keras
from keras.layers import Multiply, Average, Concatenate, Conv2D
import single_stream_model as ssm
from keras import Input
from keras import Model


conv1_params = {
    'filters': 96,
    'kernel_size': (7, 7),
    'padding': 'same',
    'strides': (2, 2),
    'batch_norm': False,
    'pooling_size': (2, 2),
    'pooling_strides': (1, 1)
}

conv1d_params = {
    'kernel_size': (1, 1),
    'padding': 'same',
    'strides': (1, 1)
}

def load_models(names, inputs):
    models = []
    for i in range(len(names)):
        models.append(ssm.create_single_stream_model(inputs[i]))

        path = "./saved_data/" + names[i] + '/weights'

        models[i].load_weights(path).expect_partial()

    return models

def get_formatted_layer_name(layer):
    idx = layer.name.rfind('_')

    if idx == -1: return layer.name

    return layer.name[:idx]

def remove_non_merged_layers(models, name_merged_layers):
    new_models = []

    for i in range(len(models)):
        names = list(map(get_formatted_layer_name, models[i].layers))

        layer_index = names.index(name_merged_layers[i])

        idx = (len(names) - layer_index) * -1
        new_model = keras.models.Sequential(models[i].layers[:idx])
        new_models.append(new_model)

    return new_models

def freeze_layers(models):
    for i in range(len(models)):
        layers = models[i].layers

        for layer in layers:
            layer.trainable = False
            
    return models

def merge_layer(type, X):
    #X is a list of [X1,X2]
    if type == 'product':
        return Multiply()(X)
    elif type == 'average':
        return Average()(X)
    elif type == 'concatenate':
        return Concatenate()(X)
    elif type == '1d':
        X1 = Concatenate()(X)
        return Conv2D(
            filters = X[0].shape[-1],
            strides = conv1d_params['strides'],
            kernel_size = conv1d_params['kernel_size']
        )(X1)


def create_two_stream_model(
        img_shape, 
        flow_shape, 
        model_type, 
        merge_on_layers = ['flatten', 'flatten'],
        model_names = ['HMDB51_image_model', 'HMDB51_flow_model'],
        numclasses = 12
    ):

    im_input = Input(shape=img_shape)
    flow_input = Input(shape=flow_shape)

    old_models = load_models(model_names, [img_shape, flow_shape])
    truncated_models = remove_non_merged_layers(old_models, merge_on_layers)
    im_model, flow_model = freeze_layers(truncated_models)

    im_activation = im_model(im_input)
    flow_activation = flow_model(flow_input)
    merged = merge_layer(model_type, [im_activation, flow_activation])
    output = ssm.dense_layers(merged, numclasses)
    return Model(inputs = [im_input, flow_input], outputs= output)