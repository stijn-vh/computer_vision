#instantiating the models, compiling and training them. using tests to generate results
import matplotlib.pyplot as plt
from models import cnn_model
from data_handler import load_mnist_data

X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_mnist_data()

#Base model
base_model = cnn_model()
history = base_model.fit(X_train, Y_train, epochs = 3)
plt.plot(history.history["accuracy"])

out = base_model.evaluate(X_test, Y_test)

print(base_model.metrics_names)
print(out)
plt.show()
#Extra convolution

#Relu activation
#relu_model = cnn_model(activation_function = 'rely')

#dropout in dense layers
#dropout = cnn_model(dropout_rate = 0.3)

#more convolutional filters of misschien exponentially weighted pooling


