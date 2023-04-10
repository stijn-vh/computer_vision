import keras
from keras.layers import Maximum, Average
from single_stream_model import create_dense_layers
from keras import Model
from keras.losses import SparseCategoricalCrossentropy

conv1_params = {
    'filters': 96,
    'kernel_size': (7, 7),
    'padding': 'same',
    'strides': (2, 2),
    'batch_norm': True,
    'pooling_size': (2, 2),
    'pooling_strides': (1, 1)
}

def load_old_models():
    return

def remove_non_merged_layers(models, name_merged_layers):
    new_models = []

    for i in range(len(models)):
        names = list(map(lambda x: x.name, models[i].layers))

        layer_index = names.index(name_merged_layers[i])

        new_model = keras.models.Sequential(models[i].layers[:-len(names)-layer_index])
        new_models.append(new_model)

    return new_models

def freeze_layers(models):
    for i in range(len(models)):
        layers = models[i].layers

        for layer in layers:
            layer.trainable = False
            
    return models

def get_merge_layers(type, models):
    last_layers = list(map(lambda model: model.layers[:-1], models))

    match type:
        case 'max':
            return Maximum(last_layers)
        case 'average':
            return Average(last_layers)
        
def create_double_stream_model(merge_layer_type):
    models = load_old_models()

    inputs = list(map(lambda model: model.input, models))

    models = remove_non_merged_layers(models, ['conv4', 'conv4'])
    models = freeze_layers(models)

    merge_layer = get_merge_layers(merge_layer_type, models)
    outputs = create_dense_layers(merge_layer)

    model = Model(inputs, outputs)
    model.compile(optimizer = 'adam',
                  loss = SparseCategoricalCrossentropy(),
                  metrics = ['accuracy'])

    return model
