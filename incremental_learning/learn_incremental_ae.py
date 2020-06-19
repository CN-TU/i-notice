#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout

import matplotlib.pyplot as plt
import sys
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import UpSampling1D, AveragePooling1D, Input, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, Reshape
import numpy as np
from keras.callbacks import TensorBoard
from time import time
import keras
import keras as K
import tensorflow
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout

import matplotlib.pyplot as plt
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, Reshape
import numpy as np
from keras.callbacks import TensorBoard
from time import time
import keras

from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import random
import keras.backend as K
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import balanced_accuracy_score
from csv import writer
from keras.layers.normalization import BatchNormalization
from keras.regularizers import Regularizer
from datetime import datetime
from keras.callbacks import ModelCheckpoint


###################################################################################

#training_order = [5,3,2,10,12,7,4,6,8,15,0,1,9,13,14]
training_order = [11,5,3,2,10,12,7,4,6,8,0,1,9,13,14] # new list without normal
#random.shuffle(training_order)
#training_order.insert(0,11) # insert normal

acc_list = []
rnd_list = []
pre_list = []
recall_list = []

seed = 1994
training_history = []
n_features = 18
n_packets = 20
n_exemplars = 5000

_train = False
evaluate = True
freeze = False

np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--nexemplars', required=True)
parser.add_argument('--train',  action='store_true')
parser.add_argument('--evaluate',  action='store_true')
parser.add_argument('--freeze', action='store_true')
parser.add_argument('--epochs_main', required=True)
parser.add_argument('--epochs_production', required=True)
parser.add_argument('--batch_size_main', required=True)
parser.add_argument('--batch_size_production', required=True)
parser.add_argument('--model_name')
args = parser.parse_args()

n_exemplars = int(args.nexemplars)
_train = args.train
evaluate = args.evaluate
freeze = args.freeze
epochs_main = int(args.epochs_main)
epochs_production = int(args.epochs_production)
batch_size_main = int(args.batch_size_main)
batch_size_production = int(args.batch_size_production)
model_name = args.model_name

###################################################################################

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def get_exemplars_initial(X_training_set, y_training_set, k):
    u = np.unique(y_training_set, axis=0)
    idx = []
    for unique in u:
        tmp = [x for x,y in enumerate(y_training_set) if (y == unique).all()]
        #if len(tmp) > k:
        	#np.random.shuffle(tmp)
            #idx.extend(list(tmp[:k]))
        #else:
            #idx.extend(list(np.random.choice(tmp, k)))
        idx.extend(tmp[:k])
    
    X_exemplars = X_training_set[idx]
    y_exemplars = y_training_set[idx]
    
    s = np.arange(y_exemplars.shape[0])
    np.random.shuffle(s)
    
    return X_exemplars[s], y_exemplars[s]

def get_exemplars_initial_herding(X_training_set, y_training_set, k):
    u = np.unique(y_training_set, axis=0)
    idx = []
    for unique in u:
        tmp = [x for x,y in enumerate(y_training_set) if (y == unique).all()]
        #if len(tmp) > k:
            #np.random.shuffle(tmp)
            #idx.extend(list(tmp[:k]))
        #else:
            #idx.extend(list(np.random.choice(tmp, k)))
        
        m = np.mean(X_training_set[tmp], (0,1))
        d = np.linalg.norm(a-b)

        idx.extend(tmp[:k])

    
    X_exemplars = X_training_set[idx]
    y_exemplars = y_training_set[idx]
    
    s = np.arange(y_exemplars.shape[0])
    np.random.shuffle(s)
    
    return X_exemplars[s], y_exemplars[s]

def get_exemplars(X_train, y_train, k, X_exemplars, y_exemplars, idx):
    np.random.shuffle(idx)
        
    #if len(idx) > k:
    #    i = idx[:k]
    #else:
    #    i = list(np.random.choice(idx, k))

    X_exemplars = np.concatenate((X_exemplars, X_train[idx[:k]]))
    #y_exemplars = np.concatenate((y_exemplars, y_train[idx[:k]]))
    
    tmp = np.zeros((y_exemplars.shape[0]+len(idx[:k]), y_exemplars.shape[1]+1))
    tmp[:y_exemplars.shape[0],:-1] = y_exemplars
    tmp[y_exemplars.shape[0]:,i+2] = 1
    y_exemplars = tmp
    
    s = np.arange(y_exemplars.shape[0])
    np.random.shuffle(s)
    
    return X_exemplars[s], y_exemplars[s]

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def b_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred, adjusted=True)

###################################################################################

X = np.load('../20/X_perfeature.npy')
y = np.load('../20/cat_uni_class.npy')

# remove normal
y = np.delete(y, 11, 1)

X = X.reshape(X.shape[0], n_packets, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

initial_indices_train = [x for x,y in enumerate(y_train) if y[training_order[0]] == 1]

#if len(initial_indices_train) > 50000:
#	initial_indices_train = random.sample(initial_indices_train, 50000)

initial_indices_train.extend([x for x,y in enumerate(y_train) if y[training_order[1]] == 1])

s = np.arange(len(initial_indices_train))
np.random.shuffle(s)
initial_indices_train = np.array(initial_indices_train)[s]

initial_indices_test = [x for x,y in enumerate(y_test) if y[training_order[0]] == 1 or y[training_order[1]] == 1]

#    initial_indices_train = random.sample(initial_indices_train, 10000)
#except Exception as e:
#    print(e)

X_training_set = X_train[initial_indices_train]
y_training_set = y_train[initial_indices_train][:, [training_order[0], training_order[1]]]

X_testing_set = X_test[initial_indices_test]
y_testing_set = y_test[initial_indices_test][:, [training_order[0], training_order[1]]]

X_exemplars, y_exemplars = get_exemplars_initial(X_training_set, y_training_set, n_exemplars)
X_exemplars.shape, y_exemplars.shape

##################################################################################

logdir = "../../autoencoder/logs/" + datetime.now().strftime("{}_%Y%m%d-%H%M%S".format(model_name))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

##################################################################################

if _train:
    # model = Sequential()
    # model.add(Conv1D(256, 16, activation='sigmoid', input_shape=(n_packets,n_features)))
    # model.add(MaxPooling1D(3))
    # model.add(Conv1D(256, 8, activation='sigmoid'))
    # model.add(MaxPooling1D(3))
    # model.add(Conv1D(64, 4, activation='sigmoid'))
    # model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.3))
    # model.add(Dense(128, activation='sigmoid'))
    # model.add(Dense(64, activation='sigmoid'))
    # model.add(Dense(32, activation='sigmoid'))
    # model.add(Dense(2, activation='softmax'))
    model = Sequential()
    model.add(Conv1D(128, 3, activation='sigmoid', input_shape=(n_packets,n_features)))
    model.add(BatchNormalization(axis=-1))
    #model.add(MaxPooling1D(3))
    #model.add(Conv1D(256, 8, activation='sigmoid'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    print(model.summary())
    #model.compile(loss = "categorical_crossentropy", optimizer ='adam', metrics=['acc' ,precision_m, recall_m])
    model.compile(loss = "binary_crossentropy", optimizer ='adam', metrics=['acc' ,precision_m, recall_m])

    verbose, epochs, batch_size = 1, epochs_main, batch_size_main

    history = model.fit(X_training_set, y_training_set, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[tensorboard_callback])

    training_history.append(history)

    model.save('models/{}.h5'.format(model_name))

else:
    #model = load_model('models/{}.h5'.format(model_name))
    AE = load_model('models/{}.h5'.format(model_name))
    AE.layers.pop()
    AE.layers.pop()
    AE.layers.pop()
    AE.layers.pop()
    #AE.layers.pop()
    #AE.layers.pop()

    last_layer = AE.get_layer('dropout_1').output

    d1 = Dense(128, activation='relu')(last_layer)
    d2 = Dense(64, activation='relu')(d1)
    d3 = Dense(32, activation='relu')(d2)
    d4 = Dense(2, activation='sigmoid')(d3)

    model = Model(input=AE.input, output=d4)

    model.compile(loss = "binary_crossentropy", optimizer ='adam', metrics=['acc' ,precision_m, recall_m])

###################################################################################

if freeze:
    model.get_layer('conv1d_1').trainable = False
    model.get_layer('batch_normalization_1').trainable = False
    model.get_layer('max_pooling1d_1').trainable = False
    model.get_layer('conv1d_2').trainable = False
    model.get_layer('batch_normalization_2').trainable = False
    model.get_layer('global_average_pooling1d_1').trainable = False
    model.get_layer('dropout_1').trainable = False

###################################################################################
# Initial training

verbose, epochs, batch_size = 1, epochs_main, batch_size_main

history = model.fit(X_training_set, y_training_set, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[tensorboard_callback])

training_history.append(history)

###################################################################################

if evaluate:
    _,acc, precision, recall = model.evaluate(X_testing_set, y_testing_set, batch_size=batch_size, verbose=verbose)
    print('##### main model acc: ',acc)
    acc_list.append(acc)
    pre_list.append(precision)
    recall_list.append(recall)

    c,u = np.unique(y_testing_set, axis=0, return_counts=True)
    rnd = np.max(u)/np.sum(u)
    rnd_list.append(rnd)

    print('###### state: {} {}'.format(acc, rnd))

###################################################################################

for i,j in enumerate(training_order[2:]):
    
    print("#####################################")

    # construct training
    initial_indices_train = [x for x,y in enumerate(y_train) if y[j] == 1]

    #if len(initial_indices_train) < n_exemplars:
    #    add = n_exemplars - len(initial_indices_train)
    #    initial_indices_train.extend(list(np.random.choice(initial_indices_train, add)))
    
    tmp = np.zeros((y_exemplars.shape[0]+len(initial_indices_train), y_exemplars.shape[1]+1))
    tmp[:y_exemplars.shape[0],:-1] = y_exemplars
    tmp[y_exemplars.shape[0]+1:,i+2] = 1
    
    X_training_set = np.concatenate((X_exemplars, X_train[initial_indices_train]))
    
    s = np.arange(tmp.shape[0])
    np.random.shuffle(s)
    
    X_training_set = X_training_set[s]
    y_training_set = tmp[s]
    
    # change model //TODO: check weights initialisation 
    n_outputs = i+2
    n_new_outputs = 1

    n_outputs += n_new_outputs
    weights = model.get_layer('dense_4').get_weights()
    shape = weights[0].shape[0]
    weights[1] = np.concatenate((weights[1], np.zeros(n_new_outputs)), axis=0)
    weights[0] = np.concatenate((weights[0], -0.01 * np.random.random_sample((shape, n_new_outputs)) + 0.01), axis=1)
    #Deleting the old output layer
    model.layers.pop()
    last_layer = model.get_layer('dense_3').output
    #New output layer
    out = Dense(n_outputs, activation='softmax', name='dense_4')(last_layer)
    model = Model(input=model.input, output=out)
    #set weights to the layer
    model.get_layer('dense_4').set_weights(weights)
    model.compile(loss = "categorical_crossentropy", optimizer ='adam', metrics=['acc' ,precision_m, recall_m])
    
    # train
    verbose, epochs, batch_size = 0, epochs_production, batch_size_production
    history = model.fit(X_training_set, y_training_set, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[tensorboard_callback])
    training_history.append(history)
    model.save('models/{}_model_{}_classes.h5'.format(model_name, i+3))
    
    # construct test
    initial_indices_test = [x for x,y in enumerate(y_test) if y[j] == 1]
    
    tmp = np.zeros((y_testing_set.shape[0]+len(initial_indices_test), y_testing_set.shape[1]+1))
    tmp[:y_testing_set.shape[0],:-1] = y_testing_set
    tmp[y_testing_set.shape[0]+1:,i+2] = 1
    
    X_testing_set = np.concatenate((X_testing_set, X_test[initial_indices_test]))
    y_testing_set = tmp
    
    # test
    _,acc, precision, recall = model.evaluate(X_testing_set, y_testing_set, batch_size=batch_size, verbose=verbose)
    print("\n", "#### Subtraining with", y_testing_set.shape[1], acc)
    acc_list.append(acc)

    pre_list.append(precision)
    recall_list.append(recall)

    c,u = np.unique(y_testing_set, axis=0, return_counts=True)
    rnd = np.max(u)/np.sum(u)
    rnd_list.append(rnd)

    print('###### state: {} {}'.format(acc, rnd))

    # construct new exemplars
    X_exemplars, y_exemplars = get_exemplars(X_train, y_train, n_exemplars, X_exemplars, y_exemplars, initial_indices_train)

    print('##### statistics')
    print('training set {}'.format(y_training_set.shape))
    print('testing set {}'.format(y_testing_set.shape))
    print('exemplars till now {}'.format(y_exemplars.shape))
    c,u = np.unique(y_exemplars, axis=0, return_counts=True)
    print('classes in exemplars {}'.format(u))
###################################################################################

print("#### Constructed final set", X_testing_set.shape, y_testing_set.shape)
print("#### Original set", X_test.shape, y_test.shape)

###################################################################################

pred = model.predict(X_testing_set)

print('#### Classification report')
print(classification_report(y_testing_set.argmax(axis=1), pred.argmax(axis=1)))

print('#### MCC')
print(matthews_corrcoef(y_testing_set.argmax(axis=1), pred.argmax(axis=1)))

print('#### Confusion Matrix')
print(confusion_matrix(y_testing_set.argmax(axis=1), pred.argmax(axis=1)))

###################################################################################

plt.figure()
plt.plot([2,3,4,5,6,7,8,9,10,11,12,13,14,15], acc_list, label='acc')
plt.plot([2,3,4,5,6,7,8,9,10,11,12,13,14,15], pre_list, label='prec')
plt.plot([2,3,4,5,6,7,8,9,10,11,12,13,14,15], recall_list, label='rec')
plt.plot([2,3,4,5,6,7,8,9,10,11,12,13,14,15], rnd_list, label='maj guessing')
training_order[1] = '{} & '.format(training_order[0]) + '{}'.format(training_order[1])
#plt.xticks(training_order[1:])
plt.ylim([0,1])
plt.xlabel('Incremental Classes')
plt.ylabel('Score Classes')
plt.legend()
plt.title('{} {} {} {} {}'.format(n_exemplars, epochs_main, epochs_production, batch_size_main, batch_size_production, training_order[1:]))
plt.savefig('{}.png'.format(model_name))

###################################################################################

append_list_as_row('acc_list.txt', acc_list)
append_list_as_row('pre_list.txt', pre_list)
append_list_as_row('recall_list.txt', recall_list)
append_list_as_row('rnd_list.txt', rnd_list)