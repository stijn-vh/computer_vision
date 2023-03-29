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

stop_epochs = [9, 9, 12, 15, 14]


def get_callbacks(model_name):
    checkpoint_path = "./saved_data/" + model_name + "/weights"
    tensor_board_path = "./saved_data/" + model_name + "/tensor_board_logs"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True,
                                                    save_weights_only=True)
    tensor_board = tf.keras.callbacks.TensorBoard(
        log_dir=tensor_board_path,
        write_images=True, write_steps_per_second=True, histogram_freq=1)
    return [checkpoint, tensor_board, hyper_params["early_stopping"], get_learning_rate_scheduler()]


def load_models(models, models_names):
    for i in range(len(models)):
        models[i].load_weights("./saved_data/" + models_names[i] + "/weights").expect_partial()


def evaluate_models(models, models_names, X_train, X_valid, Y_train, Y_valid):
    for i in range(len(models)):
        print("for ", models_names[i], " the training loss and accuracy are ", models[i].evaluate(X_train, Y_train))
        print("for ", models_names[i], " the validation loss and accuracy are ", models[i].evaluate(X_valid, Y_valid))


def train_model(model, model_name, X_train, X_valid, Y_train, Y_valid):
    print("training model ", model_name)
    callbacks = get_callbacks(model_name)
    model.fit(X_train, Y_train, epochs=hyper_params["train_epochs"], batch_size=hyper_params["batch_size"],
              validation_data=(X_valid, Y_valid), verbose=1,
              callbacks=callbacks)


def train_best(model, model_name, X_train, Y_train, epochs):
    print("training model ", model_name, "On train and val data")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="./saved_data/" + model_name + "/weights", verbose=1,
                                                    save_weights_only=True)
    tensor_board = tf.keras.callbacks.TensorBoard(
        log_dir="./saved_data/" + model_name + "/tensor_board_logs",
        write_images=True, write_steps_per_second=True, histogram_freq=1)

    model.fit(X_train, Y_train, epochs=epochs, callbacks=[checkpoint, tensor_board])


def analyse_best(model, model_name, epochs):
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_mnist_data()
    X_train = np.concatenate((X_train, X_valid))
    Y_train = np.concatenate((Y_train, Y_valid))

    if not os.path.isdir(os.path.join('./saved_data', model_name)):
        train_best(model, model_name, X_train, Y_train, epochs)
    else:
        load_models([model], [model_name])
    print(model_name, " will be evaluated on the test set")
    model_test(model, X_test, Y_test)


def train_models(models, models_names):
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_mnist_data()
    if not os.path.isdir('./saved_data'):
        os.mkdir('./saved_data')

    for i in range(len(models)):
        if os.path.isdir(os.path.join('./saved_data', models_names[i])):
            continue
        else:
            train_model(models[i], models_names[i], X_train, X_valid, Y_train, Y_valid)

    load_models(models, models_names)
    evaluate_models(models, models_names, X_train, X_valid, Y_train, Y_valid)
    return


def model_test(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(tf.nn.softmax(Y_pred, axis=-1), axis=-1)
    confusion_matrix = tf.math.confusion_matrix(Y_test, Y_pred)
    print("the test loss and accuracy are ", model.evaluate(X_test, Y_test))
    print("The confusion matrix on the test set is given by", confusion_matrix.numpy())


if __name__ == '__main__':
    models = [cnn_model(), cnn_model(extra_conv=True), cnn_model(activation_function="relu"),
              cnn_model(dropout_rate=0.3), cnn_model(batch_norm=True)]
    models_names = ["base_model", "extra_conv_model", "relu_model", "dropout_model", "batch_norm_model"]
    # train_models(models, models_names)

    combined_model = cnn_model(activation_function="relu", dropout_rate=0.3, batch_norm=True, data_augmentation=True)
    analyse_best(combined_model, "combined_model_30_aug", epochs=30)
    # analyse_best(models[2], models_names[2]+"_trainval", epochs = 12)
    # analyse_best(models[4], models_names[4]+"_trainval",  epochs = 14)
