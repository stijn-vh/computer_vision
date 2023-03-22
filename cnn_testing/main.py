# instantiating the models, compiling and training them. using tests to generate results
import matplotlib.pyplot as plt
from models import cnn_model
from data_handler import load_mnist_data
from keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from json_helper import JsonHelper

JH = JsonHelper()

EPOCHS = 15
LOAD_MODEL = False

X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_mnist_data()

saved_model_path = "saved_base_model"
history_path = "base_model_history"
callback = tf.keras.callbacks.ModelCheckpoint(filepath=saved_model_path, verbose=1, save_best_only=True)

# Base model
base_model = cnn_model()
base_model.compile(optimizer='adam',
                   loss=SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
if LOAD_MODEL:
    print("loading model")
    base_model.load_weights(saved_model_path)
    history = JH.load_from_json(history_path)
else:
    print("fitting model")
    history = base_model.fit(X_train, Y_train, epochs=EPOCHS, validation_data=(X_valid, Y_valid), verbose=1,
                             callbacks=[callback])
    JH.save_to_json(history_path, history)

plt.plot(base_model.history["accuracy"], base_model.history["val_accuracy"])
plt.plot(base_model.history["val_loss"], base_model.history["loss"])

out = base_model.evaluate(X_test, Y_test)

print(base_model.metrics_names)
print(out)
plt.show()
# Extra convolution

# Relu activation
# relu_model = cnn_model(activation_function = 'rely')

# dropout in dense layers
# dropout = cnn_model(dropout_rate = 0.3)

# more convolutional filters of misschien exponentially weighted pooling
