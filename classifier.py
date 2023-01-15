import numpy as np
from tensorflow import keras as keras
import matplotlib.pyplot as plt


def plot_history(history):
    fig, axs = plt.subplots(2)

    # accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train acc")
    axs[0].plot(history.history["val_accuracy"], label="test acc")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error evaluation")

    plt.show()


# Build the CNN Model
def build_model(shape):
    # 3 Convolutional Layers with Max-Pooling

    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  input_shape=shape))  # nr of kernels, size of kernel and activation function, input shape
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))  # max pooling layer | pool size

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  input_shape=shape))  # nr of kernels, size of kernel and activation function, input shape
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))  # max pooling layer | pool size
    model.add(keras.layers.Dropout(0.3))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  input_shape=shape))  # nr of kernels, size of kernel and activation function, input shape
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))  # max pooling layer | pool size
    model.add(keras.layers.Dropout(0.3))

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))  # add dense layer

    # output layer that uses softmax
    model.add(keras.layers.Dense(10, activation='softmax'))  # one unit for each genre

    return model


def predict_genre(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)  # X -> (130, 13, 1)
    # prediction -> 2d array

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {} | Predicted index: {}".format(y, predicted_index))


