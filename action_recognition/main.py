from data_loader import StanfordLoader, Hmdb51Loader
from models.single_stream_model import create_single_stream_model
import cv2 as cv
import numpy as np
import tensorflow as tf


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


def get_all_callbacks():
    model_names = ["stanford_image_model", "HMDB51_image_model", "HMDB51_flow_model", "two_stream_model"]
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


if __name__ == '__main__':
    print("Cuda Availability: ", tf.test.is_built_with_cuda())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print('Start loading data')

    params = {
        'image_size': (224, 224),
        'batch_size': 32,
        'print_info': True,
        'test_size': 0.1,
        'val_size': 0.1,
        'epochs': 75
    }
    SL = StanfordLoader(params)
    HL = Hmdb51Loader(params)

    S_train_gen, S_val_gen, S_test_gen = SL.get_image_generators()
    H_train_im_gen, H_val_im_gen, H_test_im_gen = HL.get_image_generators()
    H_train_flow_gen, H_val_flow_gen, H_test_flow_gen = HL.get_optical_flow_generators()
    print("Loading data finished")

    im_num =1
    cv.imshow("stanford train image", np.float32(S_train_gen.next()[0][im_num]))
    cv.imshow("HMDB train image", np.float32(H_train_im_gen.__next__()[0][im_num]))
    flowstack = np.float32(H_train_flow_gen.__next__()[0][im_num])
    cv.imshow("one optical flow from train image", draw_hsv(flowstack[:, :, 0:2]))
    cv.waitKey(0)

    print("Start training models")
    stanford_image_model_callbacks, HMDB51_image_model_callbacks, HMDB51_flow_model_callbacks, two_stream_model_callbacks = get_all_callbacks()

    im_shape = (params["image_size"][0], params["image_size"][1], 3)
    flow_shape = (params["image_size"][0], params["image_size"][1], 32)
    stanford_image_model = create_single_stream_model(im_shape)
    stanford_image_model.summary()
    print("start training stanford model")
    stanford_image_model.fit(S_train_gen, validation_data=S_val_gen, epochs=params['epochs'],
                             callbacks=stanford_image_model_callbacks, verbose=1)

    HMDB51_image_model = create_single_stream_model(im_shape)
    print("start trainingHMDB51 im model")
    HMDB51_image_model.fit(H_train_im_gen, validation_data=H_val_im_gen, epochs=params['epochs'],
                             callbacks=HMDB51_image_model_callbacks, verbose=1)

    HMDB51_flow_model = create_single_stream_model(flow_shape)
    print("start trainingHMDB51 flow model")
    HMDB51_flow_model.fit(H_train_flow_gen, validation_data=H_val_flow_gen, epochs=params['epochs'],
                             callbacks=HMDB51_flow_model_callbacks, verbose=1)

    #Note that twostream model should not augment the training HMDB51 images. Also need some way to combine data inputs simultaneously
    #The stanford_image_model_loss1.64_valacc0.46_epoch20 was trained on RGB images instead of BGR. Thus fine tuning will not work as intended.

#     Epoch 34: val_loss improved from 1.59693 to 1.57232, saving model to ./saved_data/stanford_image_model\weights
#    77/77 [==============================] - 33s 424ms/step - loss: 1.3975 - accuracy: 0.5075 - sparse_top_k_categorical_accuracy: 0.8024 - val_loss: 1.5723 - val_accuracy: 0.4745 - val_sparse_top_k_categorical_accuracy: 0.7445

# Epoch 27: val_loss improved from 1.66525 to 1.57069, saving model to ./saved_data/HMDB51_image_model\weights
# 38/38 [==============================] - 14s 367ms/step - loss: 1.4807 - accuracy: 0.5000 - sparse_top_k_categorical_accuracy: 0.7812 - val_loss: 1.5707 - val_accuracy: 0.5078 - val_sparse_top_k_categorical_accuracy: 0.7031
# echter ook Epoch 43: val_loss did not improve from 1.57069
# 38/38 [==============================] - 14s 369ms/step - loss: 1.1798 - accuracy: 0.6135 - sparse_top_k_categorical_accuracy: 0.8520 - val_loss: 1.6094 - val_accuracy: 0.5703 - val_sparse_top_k_categorical_accuracy: 0.7500
# Epoch 44/50

# Epoch 8: val_loss improved from 2.31198 to 2.05122, saving model to ./saved_data/HMDB51_flow_model\weights
# 38/38 [==============================] - 31s 807ms/step - loss: 1.9933 - accuracy: 0.3051 - sparse_top_k_categorical_accuracy: 0.6217 - val_loss: 2.0512 - val_accuracy: 0.2578 - val_sparse_top_k_categorical_accuracy: 0.5547
