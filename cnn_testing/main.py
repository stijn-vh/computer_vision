# instantiating the models, compiling and training them. using tests to generate results
import matplotlib.pyplot as plt
from models import cnn_model
from data_handler import load_mnist_data
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

import tensorflow as tf
from json_helper import JsonHelper
import os

hyper_params = {
    "train_epochs": 15,
    "batch_size": 32,
    "early_stopping": tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=15,
                                                       verbose=1,
                                                       restore_best_weights=True)
}

current_lr = 1
count = 0
def learn_rate_scheduler(epoch):
    global count, current_lr

    if count == 5:
        current_lr / 2
        count = 0
    else:
        count += 1
    
    return float(current_lr)

def get_learning_rate_scheduler():
    return LearningRateScheduler(learn_rate_scheduler)

def get_callbacks(model_name):
    checkpoint_path = "./saved_data/" + model_name + "/weights"
    tensor_board_path = "./saved_data/" + model_name + "/tensor_board_logs"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True,
                                                    save_weights_only=True)
    tensor_board = tf.keras.callbacks.TensorBoard(
        log_dir=tensor_board_path,
        write_images=True, write_steps_per_second=True, histogram_freq=1)
    return [checkpoint, tensor_board, hyper_params["early_stopping"], get_learning_rate_scheduler()]


def train_new_model(model, model_name, X_train, X_valid, Y_train, Y_valid):
    print("training model ", model_name)
    callbacks = get_callbacks(model_name)
    model.fit(X_train, Y_train, epochs=hyper_params["train_epochs"], batch_size=hyper_params["batch_size"], validation_data=(X_valid, Y_valid), verbose=1,
              callbacks=callbacks)


def train_models(models, models_names):
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_mnist_data()
    if not os.path.isdir('./saved_data'):
        os.mkdir('./saved_data')

    for i in range(len(models)):
        if os.path.isdir(os.path.join('./saved_data', models_names[i])):
            continue
        train_new_model(models[i], models_names[i], X_train, X_valid, Y_train, Y_valid)
    return


if __name__ == '__main__':
    models = [cnn_model(), cnn_model(extra_conv=True), cnn_model(activation_function="relu"),
              cnn_model(dropout_rate=0.3), cnn_model(batch_norm=True)]

    models_names = ["base_model", "extra_conv_model", "relu_model", "dropout_model", "batch_norm_model"]

    train_models(models, models_names)
