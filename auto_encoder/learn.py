# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2020, Institute of Telecommunications, TU Wien
#
# Name        : learn.py
# Description : an autoencoder architechture to train the feature extractor of iNotice
# Author      : Fares Meghdouri
#
# Notes       : 
#
# Change-log  : 19-06-2020: //FM// Initial commit
#
# TODOs       : outsource variables more
#
#******************************************************************************

import tensorflow
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
import matplotlib.pyplot as plt
from keras.layers import UpSampling1D, AveragePooling1D, Input, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, Reshape
import numpy as np
import keras
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

#******************************************************************************

epochs = 128							# number of training epochs 
batch_size = 8							# batch size
freeze = True							# freeze the feature extractor weights 
output_model = 'trained_FE_and_FCL.h5'  # where to save the final model
mawi_data_path = 'mawi.npy'				# mawi data used for training
training_data_path = 'X_train.npy'		# cicids17 data
training_labels_path = 'y_train.npy'    # cicids17 labels
_plot = 'loss_cnnae.png'				# name of output loss plot

#******************************************************************************

def read_data():
	return np.load(mawi_data_path), np.load(training_data_path), np.load(training_labels_path)

#******************************************************************************

def main():

	X_mawi, X_train, y_train = read_data()

	# TRAIN THE AUTOENCODER
	input_sig = Input(batch_shape=(None,20,18))
	x1 = Conv1D(128, 3, activation='sigmoid')(input_sig)
	x2 = BatchNormalization(axis=-1)(x1)
	x3 = MaxPooling1D(3)(x2)
	x4 = Conv1D(64, 3, activation='relu')(x3)
	x5 = BatchNormalization(axis=-1)(x4)
	x6 = GlobalAveragePooling1D()(x5)
	x7 = Dropout(0.5)(x6)
	encoded = Dense(64, activation='relu')(x7)

	d2 = Dense(64, activation='sigmoid')(encoded)
	d3 = Dense(128, activation='tanh')(d2)
	d4 = Dense(200,activation='linear')(d3)
	d5 = Reshape((20,10))(d4)
	decoded = Conv1D(18,1,activation=None)(d5)

	AE= Model(input_sig, decoded)
	AE.compile(optimizer='adadelta', loss='mse')

	#print(model.summary())

	# save the feature extractor for future use
	checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
	history = AE.fit(X_mawi, X_mawi, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[checkpoint])

	# plot the loss
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(_plot) 

	#******************************************************************************
	# MOUNT THE FEATURE EXTRACTOR IN INOTICE (example)
	for _layer in range(6):
		AE.layers.pop()

	last_layer = AE.get_layer('dropout_1').output

	d1 = Dense(128, activation='relu')(last_layer)
	d2 = Dense(64, activation='relu')(d1)
	d3 = Dense(32, activation='relu')(d2)
	d4 = Dense(2, activation='sigmoid')(d3)

	model = Model(input=AE.input, output=d4)
	model.compile(loss = "binary_crossentropy", optimizer ='adam', metrics=['acc' ])

	# if we want to finetune only the fully connected layers
	if freeze:
		model.get_layer('conv1d_1').trainable = False
		model.get_layer('batch_normalization_1').trainable = False
		model.get_layer('max_pooling1d_1').trainable = False
		model.get_layer('conv1d_2').trainable = False
		model.get_layer('batch_normalization_2').trainable = False
		model.get_layer('global_average_pooling1d_1').trainable = False
		model.get_layer('dropout_1').trainable = False

	model.fit(X_train, y_train, validation_split=0.33, epochs=2, batch_size=8, verbose=1)
	model.save(output_model)

if __name__ == "__main__":
    main()