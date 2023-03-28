def make_plots(histories, plot_titles, stop_epochs):
    stop_epochs = np.array(stop_epochs) - 1  # Because plotting is from epoch 0 to 14 instead of 1 to 15
    for i in range(len(histories)):
        plt.figure()
        plt.plot(histories[i]["accuracy"], "b", label="training accuracy")
        plt.plot(histories[i]["val_accuracy"], "r", label="validation accuracy")
        plt.vlines(stop_epochs[i], 0.75, 1, "g", label="early stopping")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.title(plot_titles[i])
        plt.show()

        plt.figure()
        plt.plot(histories[i]["loss"], "b", label="training loss")
        plt.plot(histories[i]["val_loss"], "r", label="validation loss")
        plt.vlines(stop_epochs[i], 0.75, 0, "g", label="early stopping")
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title(plot_titles[i])
        plt.show()


def load_and_plot_models():
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_mnist_data()
    print("loading models")
    plot_titles = ["base_model", "dropout_model", "relu_model", "extra_conv_model", "batch_norm_model"]
    stop_epochs = [8, 15, 13, 9, 10]
    base_model = cnn_model()
    dropout_model = cnn_model(dropout_rate=0.3)
    relu_model = cnn_model(activation_function="relu")
    extra_conv_model = cnn_model(extra_conv=True)
    batch_norm_model = cnn_model(batch_norm=True)
    models = [base_model, dropout_model, relu_model, extra_conv_model, batch_norm_model]
    model_paths = ["./saved_data/base_model", "./saved_data/dropout_model", "./saved_data/relu_model",
                   "./saved_data/extra_conv_model", "./saved_data/batch_norm_model"]
    history_paths = ["./saved_data/base_history", "./saved_data/dropout_history", "./saved_data/relu_history",
                     "./saved_data/extra_conv_history", "./saved_data/batch_norm_history"]

    histories = []
    for i in range(len(models)):
        models[i].load_weights(model_paths[i]).expect_partial()
        print(plot_titles[i], "training loss and accuracy are ", models[i].evaluate(X_train, Y_train))
        print(plot_titles[i], "validation loss and accuracy are ", models[i].evaluate(X_valid, Y_valid))
        histories.append(JH.load_pickle_object(history_paths[i]))

    make_plots(histories, plot_titles, stop_epochs)
