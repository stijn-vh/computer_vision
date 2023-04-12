import keras
from keras.layers import Maximum, Average, Concatenate, Conv2D
from single_stream_model import create_dense_layers
from keras import Model, models
from keras.losses import SparseCategoricalCrossentropy
import os
from single_stream_model import create_single_stream_model

conv1_params = {
    'filters': 96,
    'kernel_size': (7, 7),
    'padding': 'same',
    'strides': (2, 2),
    'batch_norm': True,
    'pooling_size': (2, 2),
    'pooling_strides': (1, 1)
}

conv1d_params = {
    'kernel_size': (1, 1),
    'padding': 'same',
    'strides': (1, 1)
}

def load_models(names, inputs):
    dirname = os.path.dirname(__file__)
    dirname = dirname.replace('models', '')
    path_prefix = os.path.join(dirname, 'data\saved_data_cv5\\')

    models = []
    for i in range(len(names)):
        models.append(create_single_stream_model(inputs[i]))

        path = path_prefix + names[i] + '/weights'

        models[i].load_weights(path).expect_partial()

    return models

def remove_non_merged_layers(models, name_merged_layers):
    new_models = []

    for i in range(len(models)):
        names = list(map(lambda x: x.name, models[i].layers))

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

def get_merge_layers(type, models):
    outputs = list(map(lambda model: model.output[:-1], models))

    match type:
        case 'max':
            return Maximum()(outputs)
        case 'average':
            return Average()(outputs)
        case 'concantate':
            return Concatenate()(outputs)
        case '1d':
            x = Concatenate()(outputs)

            return Conv2D(
                filters = outputs[0].shape[-1],
                strides = conv1d_params['strides'],
                kernel_size = conv1d_params['kernel_size']
            )(x)

def create_two_stream_model(
        img_shape, 
        flow_shape, 
        model_type, 
        merge_on_layers = ['flatten', 'flatten_1'], 
        model_names = ['HMDB51_image_model', 'HMDB51_flow_model']
    ):
    old_models = load_models(model_names, [img_shape, flow_shape])

    inputs = list(map(lambda model: model.input, old_models))

    truncated_models = remove_non_merged_layers(old_models, merge_on_layers)
    truncated_models = freeze_layers(truncated_models)

    merge_layer = get_merge_layers(model_type, truncated_models)
    outputs = create_dense_layers(merge_layer, 12)

    return Model(inputs, outputs)