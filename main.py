# build CNN
from tensorflow import keras as keras
from visualisation import *
from classifier import plot_history, build_model
from data_processing import prepare_datasets

if __name__ == "__main__":
    # plot_waveform()
    # plot_mfcc()
    # plot_one_waveform()
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.3,
                                                                                    0.4)  # x- inpput y - output / target

    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # extract input shape
    built_model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    built_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the CNN
    history = built_model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=300)

    # evaluate the CNN on the test set
    test_eror, test_accuracy = built_model.evaluate(X_test, y_test, verbose=1)
    print("Acc on test set is: {}".format(test_accuracy))

    plot_history(history)

    # make prediction on a sample
    # X = X_test[10]
    # y = y_test[10]
    # predict_genre(built_model, X, y)
