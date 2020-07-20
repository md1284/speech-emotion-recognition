"""
This example demonstrates how to use `CNN` model from
`speechemotionrecognition` package
"""
from keras.utils import np_utils

import numpy as np
from common import extract_data
from speechemotionrecognition.dnn import CNN, LSTM
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


def cnn_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    model = CNN(input_shape=x_train[0].shape,
                num_classes=num_labels)
    
    str = '../models/best_model_CNN.h5'
    model.load_model(str)

    print('load success')
    model.trained = True
    #model.train(x_train, y_train, x_test, y_test_train)
    #model.evaluate(x_test, y_test)


    filename = '../dataset/Neutral/09b03Nb.wav'
    mfcc = get_feature_vector_from_mfcc(filename, flatten=to_flatten)
    print(mfcc.shape)
    mfcc = np.array([mfcc])
    print(mfcc.shape)
    mfcc = mfcc.reshape(mfcc.shape[0], mfcc.shape[1], mfcc.shape[2], 1)
    #mfcc = mfcc.reshape(in_shape[0], in_shape[1], 1)
    mfcc = model.model.predict(mfcc)
    print(mfcc)
    print(np.argmax(mfcc))

    #print('prediction', model.predict_one(mfcc),
    #      'Actual 3')
    print('CNN Done')



def lstm_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape,
                 num_classes=num_labels)
    
    str = '../models/best_model_LSTM.h5'
    model.load_model(str)

    print('load success')
    model.trained = True
    #model.train(x_train, y_train, x_test, y_test_train, n_epochs=50)
    #model.evaluate(x_test, y_test)


    filename = '../dataset/Neutral/09b03Nb.wav'
    mfcc = get_feature_vector_from_mfcc(filename, flatten=to_flatten)
    mfcc = np.array([mfcc])
    mfcc = model.model.predict(mfcc)


    print('prediction', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
          'Actual 3')


if __name__ == "__main__":
	cnn_example()
	lstm_example()
