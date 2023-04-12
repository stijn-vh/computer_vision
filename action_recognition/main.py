from data_loader import StanfordLoader, Hmdb51Loader
from double_stream_model import create_two_stream_model
from single_stream_model import create_single_stream_model


import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # rescale max value to 255
    maxvalue = np.max(bgr)
    bgr = (255 // maxvalue) * bgr
    return bgr


def show_data(S_gen, H_im_gen, H_flow_gen):
    im_num = 1
    cv.imshow("stanford train image", np.float32(S_gen.next()[0][im_num]))
    cv.imshow("HMDB train image", np.float32(H_im_gen.__next__()[0][im_num]))
    flowstack = np.float32(H_flow_gen.__next__()[0][im_num])
    cv.imshow("one optical flow from train image", draw_hsv(flowstack[:, :, 0:2]))
    cv.waitKey(0)


def get_all_callbacks(model_names):
    checkpoint_paths = ["./saved_data/" + model_name + "/weights" for model_name in model_names]
    tensorboard_paths = ["./saved_data/" + model_name + "/tensorboard_logs" for model_name in model_names]
    checkpoints = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True,
                                                      save_weights_only=True) for checkpoint_path in checkpoint_paths]
    tensor_boards = [tf.keras.callbacks.TensorBoard(
        log_dir=tensor_board_path,
        write_images=True, write_steps_per_second=True, histogram_freq=1) for tensor_board_path in tensorboard_paths]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=20,
                                                      verbose=1,
                                                      restore_best_weights=True)
    callbacks = []
    for model_num in range(len(model_names)):
        callbacks.append([checkpoints[model_num], tensor_boards[model_num], early_stopping])
    return callbacks


def train_model(model, callback, epochs, train_gen, val_gen):
    model.fit(train_gen, validation_data=val_gen, epochs=epochs,
              callbacks=callback, verbose=1)
    return model


def evaluate_model(model, train_gen, val_gen, test_gen):
    print("train data performance:", model.evaluate(train_gen))
    print("validation data performance:", model.evaluate(val_gen))
    print("test data performance:", model.evaluate(test_gen))


def train_image_models(im_shape, epochs,last_layer_epochs, fine_tune_epochs, fine_tune_lr, retrain_stanford):
    image_model = create_single_stream_model(im_shape)
    print("The image model architecture:")
    image_model.summary()
    if retrain_stanford:
        print("start training image model with stanford data")
        S_train_gen, S_val_gen, S_test_gen = SL.get_image_generators()
        image_model.compile(optimizer=Adam(),
                            loss=SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])
        image_model = train_model(image_model, stanford_image_model_callbacks, epochs, S_train_gen, S_val_gen)
    else:
        image_model.load_weights("./saved_data/stanford_image_model/weights").expect_partial()

    print("start fine tuning stanford image model with HMDB51 data")
    H_train_im_gen, H_val_im_gen, H_test_im_gen = HL.get_image_generators()

    print("retraining the last dense layer first")
    for layer in image_model.layers:
        layer.trainable = False
    image_model.layers[-2].trainable = True
    image_model.compile(optimizer=Adam(),
                        loss=SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])
    image_model = train_model(image_model, HMDB51_image_model_callbacks, last_layer_epochs, H_train_im_gen, H_val_im_gen)

    print("fine tuning the entire model")
    for layer in image_model.layers:
        layer.trainable = True
    image_model.compile(optimizer=Adam(fine_tune_lr),
                        loss=SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])
    train_model(image_model, HMDB51_image_model_callbacks, fine_tune_epochs, H_train_im_gen, H_val_im_gen)


def train_flow_model(flow_shape, epochs):
    H_train_flow_gen, H_val_flow_gen, H_test_flow_gen = HL.get_optical_flow_generators()
    flow_model = create_single_stream_model(flow_shape)
    flow_model.compile(optimizer=Adam(),
                       loss=SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])
    print("start training HMDB51 flow model")
    train_model(flow_model, HMDB51_flow_model_callbacks, epochs, H_train_flow_gen, H_val_flow_gen)


def train_two_stream_model(im_shape, flow_shape, epochs, callback, model_type):
    # Makes sure the images and flows are not augmented or shuffled for the dual learning.
    H_train_im_gen, H_val_im_gen, H_test_im_gen = HL.get_image_generators(use_data_augmentation=False, shuffle=False)
    H_train_flow_gen, H_val_flow_gen, H_test_flow_gen = HL.get_optical_flow_generators(shuffle=False)
    two_stream_model = create_two_stream_model(im_shape, flow_shape, model_type)
    two_stream_model.compile(optimizer=Adam(),
                             loss=SparseCategoricalCrossentropy(),
                             metrics=['accuracy'])
    print("the two stream model architecture is the following:")
    two_stream_model.summary()
    train_model(two_stream_model, callback, epochs, combine_generators(H_train_im_gen, H_train_flow_gen),
                combine_generators(H_val_im_gen, H_val_flow_gen))

def combine_generators(gen1, gen2):
    while True:
        x1, y1 = gen1.__next__()
        x2, y2 = gen2.__next__()
        yield [x1, x2], y1

def train_all_two_stream_models(im_shape, flow_shape, epochs):
    train_two_stream_model(im_shape, flow_shape, epochs, two_stream_model_avg_callbacks, "average")
    train_two_stream_model(im_shape, flow_shape, epochs, two_stream_model_max_callbacks, "max")
    train_two_stream_model(im_shape, flow_shape, epochs, two_stream_model_conc_callbacks, "concatenate")
    train_two_stream_model(im_shape, flow_shape, epochs, two_stream_model_1d_callbacks, "1d")


def load_evaluate_models(model_names, params):
    H_train_im_gen, H_val_im_gen, H_test_im_gen = HL.get_image_generators(use_data_augmentation=False, shuffle=False)
    H_train_flow_gen, H_val_flow_gen, H_test_flow_gen = HL.get_optical_flow_generators(shuffle=False)
    #This could have been simplified by loading/saving the entire models instead of just the weights
    for name in model_names:
        if name == "stanford_image_model":
            train_gen, val_gen, test_gen = SL.get_image_generators(use_data_augmentation=True)
            model = create_single_stream_model(params['im_shape'])
        elif name == "HMDB51_image_model":
            train_gen, val_gen, test_gen = H_train_im_gen, H_val_im_gen, H_test_im_gen
            model = create_single_stream_model(params['im_shape'])
        elif name == "HMDB51_flow_model":
            train_gen, val_gen, test_gen = H_train_flow_gen, H_val_flow_gen, H_test_flow_gen
            model = create_single_stream_model(params['flow_shape'])
        else:
            train_gen, val_gen, test_gen =  [H_train_im_gen,H_train_flow_gen], [H_val_im_gen, H_val_flow_gen], [H_test_im_gen, H_test_flow_gen]
            if name == "two_stream_model_avg":
                model = create_two_stream_model(params['im_shape'], params['flow_shape'], "average")
            elif name == "two_stream_model_max":
                model = create_two_stream_model(params['im_shape'], params['flow_shape'], "max")
            elif name == "two_stream_model_conc":
                model = create_two_stream_model(params['im_shape'], params['flow_shape'], "concatenate")
            elif name == "two_stream_model_1d":
                model = create_two_stream_model(params['im_shape'], params['flow_shape'], "1d")
            else:
                raise Exception("invalid model name provided")
            

        print("model name", name, "has summary")
        model.summary()


        model.compile(
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
        model.load_weights("./saved_data/" + name + "/weights").expect_partial()
        print("performance for ", name)
        evaluate_model(model, train_gen, val_gen, test_gen)


if __name__ == '__main__':
    model_names = ["stanford_image_model", "HMDB51_image_model", "HMDB51_flow_model", "two_stream_model_avg","two_stream_model_max","two_stream_model_conc","two_stream_model_1d" ]
    stanford_image_model_callbacks, HMDB51_image_model_callbacks, HMDB51_flow_model_callbacks, two_stream_model_avg_callbacks, two_stream_model_max_callbacks, two_stream_model_conc_callbacks, two_stream_model_1d_callbacks = get_all_callbacks(
        model_names)

    params = {
        'image_size': (224, 224),
        'batch_size': 32,
        'print_info': True,
        'test_size': 0.1,
        'val_size': 0.1,
        'epochs': 75,
        'last_layer_epochs': 10,
        'fine_tune_epochs': 20,
        'fine_tune_lr': 1e-3
    }
    params['im_shape'] = (params["image_size"][0], params["image_size"][1], 3)
    params['flow_shape'] = (params["image_size"][0], params["image_size"][1], 32)
    SL = StanfordLoader(params)
    HL = Hmdb51Loader(params)

    #train_image_models(params["im_shape"], params["epochs"], params["last_layer_epochs"], params['fine_tune_epochs'], params['fine_tune_lr'],retrain_stanford=False)
    # train_flow_model(params["flow_shape"], params["epochs"])
    #train_all_two_stream_models(params["im_shape"], params["flow_shape"], params["epochs"])

    load_evaluate_models(model_names, params)

    # Note that twostream model should not augment the training HMDB51 images. Also need some way to combine data inputs simultaneously
    # The stanford_image_model_loss1.64_valacc0.46_epoch20 was trained on RGB images instead of BGR. Thus fine tuning will not work as intended.

#     Epoch 34: val_loss improved from 1.59693 to 1.57232, saving model to ./saved_data/stanford_image_model\weights
#    77/77 [==============================] - 33s 424ms/step - loss: 1.3975 - accuracy: 0.5075 - sparse_top_k_categorical_accuracy: 0.8024 - val_loss: 1.5723 - val_accuracy: 0.4745 - val_sparse_top_k_categorical_accuracy: 0.7445

# Epoch 27: val_loss improved from 1.66525 to 1.57069, saving model to ./saved_data/HMDB51_image_model\weights
# 38/38 [==============================] - 14s 367ms/step - loss: 1.4807 - accuracy: 0.5000 - sparse_top_k_categorical_accuracy: 0.7812 - val_loss: 1.5707 - val_accuracy: 0.5078 - val_sparse_top_k_categorical_accuracy: 0.7031
# echter ook Epoch 43: val_loss did not improve from 1.57069
# 38/38 [==============================] - 14s 369ms/step - loss: 1.1798 - accuracy: 0.6135 - sparse_top_k_categorical_accuracy: 0.8520 - val_loss: 1.6094 - val_accuracy: 0.5703 - val_sparse_top_k_categorical_accuracy: 0.7500
# Epoch 44/50

# Epoch 8: val_loss improved from 2.31198 to 2.05122, saving model to ./saved_data/HMDB51_flow_model\weights
# 38/38 [==============================] - 31s 807ms/step - loss: 1.9933 - accuracy: 0.3051 - sparse_top_k_categorical_accuracy: 0.6217 - val_loss: 2.0512 - val_accuracy: 0.2578 - val_sparse_top_k_categorical_accuracy: 0.5547
