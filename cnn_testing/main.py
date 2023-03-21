#instantiating the models, compiling and training them. using tests to generate results
import matplotlib.pyplot as plt
from models import cnn_model
from data_handler import load_mnist_data

X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_mnist_data()

#Base model
base_model = cnn_model()
history = base_model.fit(X_train, Y_train, epochs = 10)
plt.plot(history.history["accuracy"])

#Extra convolution

#Relu activation

#dropout in dense layers

#more convolutional filters of misschien exponentially weighted pooling


