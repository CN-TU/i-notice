# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Name        : learn.py
# Description : baseclassifier of the iNotice framework
# Author      : Fares Meghdouri
#
# Notes       : 
#
# Change-log  : 18-06-2020: //FM// Initial commit
#               19-06-2020: //FM// finish the code cleanup
#
# TODOs       : outsource variables more
#
#******************************************************************************

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, Reshape
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
import matplotlib.pyplot as plt
import sys
import keras
from keras.optimizers import Adam

#******************************************************************************

parser = argparse.ArgumentParser()
parser.add_argument('--vector', default='baseline', choices=['baseline', 'wPorts', 'ipsec'], help='which representation to use (default baseline)')
parser.add_argument('--sequence_length', default='20', help='number of packets in the input sequence (default 20)')
parser.add_argument('--mode', default='binary', choices=['binary', 'multiclass'], help='binary or multiclass classification (default binary)')
parser.add_argument('--training_data', help='source path of training data', metavar='IN_PATH')
parser.add_argument('--training_labels', help='source path of training labels', metavar='IN_PATH')
parser.add_argument('--test_data', help='source path of testing data', metavar='IN_PATH')
parser.add_argument('--test_labels', help='source path of testing labels', metavar='IN_PATH')
parser.add_argument('--predict', help='make predictions from test data', metavar='OUT_PATH')

parser.add_argument('--train', help='train a new model', action='store_true')
parser.add_argument('--test', help='test the model', action='store_false')
parser.add_argument('--split_data', help='split the training data into training and testing propotions', action='store_false')

parser.add_argument('--load_model', help='load a pretrained model', metavar='IN_PATH')
parser.add_argument('--store_model', help='save the model after training', metavar='OUT_PATH')

parser.add_argument('--verbose', default=1, type=int, help='verbose level (default 1)')
parser.add_argument('--epochs', default=128, type=int, help='number of epochs (default 128)')
parser.add_argument('--batch_size', default=32, type=int, help='batch size during training (default 32)')

parser.add_argument('--plot', help='generate plots of accuracy and loss', metavar='OUT_PATH')

args = parser.parse_args()

#******************************************************************************

seed = 2020

X_train_path = args.training_data
y_train_path = training_labels
X_test_path = test_data
y_test_path = training_labels

_predict = args.predict

_train = args.train
_test = args.test
split_data = args.split_data

mode = args.mode

vector = args.vector
sequence_length = args.sequence_length
_long = True if sequence_length > 999 else False

vector_mapping = {'baseline':18, 'wPorts': 16, 'ipsec':8}

verbose = args.verbose
epochs = args.epochs
batch_size = args.batch_size

_load_model = args.load_model
_save_model = args.save_model

_plot = args.plot

#******************************************************************************

# define all models

def _binary(n_packets, n_features, loss = "categorical_crossentropy", optimizer ='adam', metrics=['accuracy'], _long=False):
    model = Sequential()
    model.add(Conv1D(128, 3, activation='sigmoid', input_shape=(n_packets,n_features)))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(3))
    if _long:
        model.add(Conv1D(128, 8, activation='sigmoid'))
        model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='sigmoid'))
    model.add(BatchNormalization(axis=-1))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss = loss, optimizer = optimizer, metrics=metrics)
    return model

def _multiclass(n_packets, n_features, loss = "binary_crossentropy", optimizer ='adam', metrics=['accuracy'], _long=False):
    model = Sequential()
    model.add(Conv1D(128, 3, activation='sigmoid', input_shape=(n_packets,n_features)))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(3))
    if _long:
        model.add(Conv1D(64, 8, activation='sigmoid'))
        model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='sigmoid'))
    model.add(BatchNormalization(axis=-1))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(15, activation='sigmoid'))
    model.compile(loss = loss, optimizer = optimizer, metrics=metrics)
    return model

def read_data():
    X = None
    y = None
    X_test = None
    y_test = None
    if _train:
        X = np.load(X_train_path, allow_pickle=True)
        y = np.load(y_train_path, allow_pickle=True)
        assert X[0].shape == (sequence_length, vector_mapping[vector]), "WARNING: The shape of the training data is not correct, it should be (:,{},{})".format(sequence_length, vector_mapping[vector])
    if _test:
        X_test = np.load(X_test_path, allow_pickle=True)
        assert X_test[0].shape == (sequence_length, vector_mapping[vector]), ""
        try:
            y_test = np.load(y_train_path, allow_pickle=True)
        except:
            pass
    return X, X_test, y, y_test

def make_plot(history, _name):
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}_acc.png'.format(_name))
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}_loss.png'.format(_name)) 

#******************************************************************************

def main():
    X_train, X_test, y_train, y_test = read_data()

    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=seed)

    if _train:
        print('INFO: Start training')
        if not X_train and not y_train:
            print("WARNING: No training data or labels...Exiting!")
            exit(1)

        if _load_model:
            model = load_model(_load_model)
        else:
            if mode == 'binary':
                model = _binary(sequence_length, vector_mapping[vector], _long=_long)

            if mode == 'multiclass':
                model = _multiclass(sequence_length, vector_mapping[vector], _long=_long)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
            
        if X_test and y_test:
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[es])
        else:
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[es])

        if _save_model:
            model.save(_save_model)
            print('INFO: Model trained and saved under {}'.format(_save_model))

        if _plot:
            make_plot(history, _plot)

    if _test:
        print('INFO: Start testing')
        if _predict:
            if X_test:
                pred = model.predict(Xnew)
                np.save('', pred)
                print('INFO: Predictions saved under {}'.format(_predict))
            else:
                print("WARNING: No testing data...Exiting!")
                exit(1)
        else:
            if X_test and y_test:
                # do something
                _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
                print('INFO: Accuracy - {}'.format(accuracy))
            else:
                print("WARNING: No testing data or labels...Exiting!")
                exit(1)


if __name__ == "__main__":
    main()