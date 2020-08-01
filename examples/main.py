'''
strates how to use `CNN` model from
`speechemotionrecognition` package
'''
from keras.utils import np_utils

import socket as sk
import numpy as np
from common import extract_data
from dnn_test import CNN, LSTM
#from speechemotionrecognition.utilities import get_feature_vector_from_mfcc
from utilities_test import get_feature_vector_from_mfcc

HOST = '192.168.1.125'
PORT = 1986
client_socket = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
client_socket.connect((HOST, PORT))

def lstm_example():
    to_flatten = False
    in_shape = np.zeros((198,39))
    model = LSTM(input_shape=in_shape.shape, num_classes=7)

    load_path = 'korean_LSTM_best_model.h5'
    model.load_model(load_path)
    model.trained = True

    '''
    filename = 'angry_test.wav'
    print('prediction angry ', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)))

    filename = 'disappoint_test.wav'
    print('prediction disappoint ', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)))

    filename = 'fear_test.wav'
    print('prediction fear ', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)))

    filename = 'happy_test.wav'
    print('prediction happy ', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)))

    filename = 'neutral_test.wav'
    print('prediction neutral ', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)))

    filename = 'sad_test.wav'
    print('prediction sad ', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)))

    filename = 'surrender_test.wav'
    print('prediction surrender ', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)))
    '''

    print('start')
    while(1):
        emotion_data = model.predict_one(
            (get_feature_vector_from_mfcc('stream', flatten=to_flatten)))
        print('prediction inputdata ', emotion_data)
        emotion_data = str(emotion_data)
        client_socket.sendall(emotion_data.encode())

    '''
    data_path = '../testset'
    cur_dir = os.getcwd()
    sys.stderr.write('curdir: %s\n' % cur_dir)
    os.chdir(data_path)

    for filename in os.listdir('.'):
        filepath = os.getcwd() + '/' + filename
        print(filename, model.predict_one(
            get_feature_vector_from_mfcc(filename, flatten=to_flatten)))
    '''

if __name__ == "__main__":
    #cnn_example()
    lstm_example()
