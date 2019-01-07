#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:14:17 2018

@author: alvaro
"""

#%%
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd


#%%

#
# Get dataset 20Newsgroups from scikit-learn
#

raw_data_train = fetch_20newsgroups(subset='train', random_state=42)
raw_data_test  = fetch_20newsgroups(subset='test',  random_state=42)

labels = raw_data_train.get('target_names')
def to_label(label):
    '''Convert numeric to text label'''
    return labels[l]

raw_x_train = raw_data_train.get('data')
raw_x_test  = raw_data_test.get('data')

labels_train = raw_data_train.get('target')
labels_test  = raw_data_test.get('target')


#%%

#
# DEFINITIONS
#

MAX_WORDS = 10000
MAX_LENGTH = 1000
EMBEDDING_DIM = 100

CNN_FILTERS_L1 = 64
CNN_LENGTH_1 = 3
CNN_LENGTH_1 = 5
CNN_LENGTH_2 = 2

N_CATEGORIES = 20

EMBEDDING_PRELOAD = True
EMBEDDING_TRAIN = True

#%%

#
# Tokenize
#

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(raw_x_train)
print(f'Found {len(tokenizer.word_index)} different words but we limit them to the top {MAX_WORDS}.')

sequences_train = tokenizer.texts_to_sequences(raw_x_train)
sequences_test  = tokenizer.texts_to_sequences(raw_x_test)

#%%
# Reverse token index
reversed_word_index = dict( zip( tokenizer.word_index.values(),
                                 tokenizer.word_index.keys()))

#%%
#
# Pad sequences and input/output matrices
#

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(sequences_train, maxlen=MAX_LENGTH)
x_test  = pad_sequences(sequences_test,  maxlen=MAX_LENGTH)

y_train = to_categorical( np.asarray(labels_train), N_CATEGORIES)
y_test  = to_categorical( np.asarray(labels_test), N_CATEGORIES)

#%%
#
# Load Glove embeddings - 100 dimensions
#
import os

glove_dir = '/Users/alvaro/Downloads/imdb/glove.6B'

embeddings_index = {}

with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        line_as_list = line.split()
        token = line_as_list[0]
        coords = np.asarray( line_as_list[1:], dtype='float32' )
        embeddings_index[token] = coords

print( f'Found {len(embeddings_index)} vectors.' )

#%%
#
# Embeddings matrix
#
embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))

for word, idx in tokenizer.word_index.items():
    if idx < MAX_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector


#%%
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense
#
# Model 1
#

model = Sequential()

# Embeddings
model.add( Embedding(MAX_WORDS, EMBEDDING_DIM, input_length = MAX_LENGTH) )

# Preload embeddings (Glove)
if EMBEDDING_PRELOAD:
    model.layers[0].set_weights( [embedding_matrix] )
# Do we have to train embedding layer?
model.layers[0].trainable = EMBEDDING_TRAIN

# CNN
model.add( Conv1D(CNN_FILTERS_L1, CNN_LENGTH_1, activation='relu', padding='same') )
#model.add( MaxPool1D(10) )
model.add( Conv1D(CNN_FILTERS_L1, CNN_LENGTH_2, activation='relu', padding='same') )
model.add( GlobalMaxPool1D())

# Dense
model.add( Dense(20, activation='softmax') )

model.summary()

#%%

#%%
#
# Compile and train
#
from keras.callbacks import ModelCheckpoint

model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['acc']
              )

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

history = model.fit(x_train, y_train,
                    epochs= 10,
                    batch_size = 128,
                    validation_split = 0.2,
                    callbacks = [checkpointer]
                    )

#%%

#%%
#Plotting results
import matplotlib.pyplot as plt

accuracies =  history.history['acc']
accuracies_val = history.history['val_acc']
indexes = range(len(accuracies))

plt.plot(indexes, accuracies, "bo", label="Training")
plt.plot(indexes, accuracies_val, "b", label="Validation")
plt.title("Accuracy - Epoch")
plt.legend()


#%%
# Model 2 (window 1)

model2 = Sequential()

# Embeddings
model2.add( Embedding(MAX_WORDS, EMBEDDING_DIM, input_length = MAX_LENGTH) )

# Preload embeddings (Glove)
if EMBEDDING_PRELOAD:
    model2.layers[0].set_weights( [embedding_matrix] )
# Do we have to train embedding layer?
model2.layers[0].trainable = EMBEDDING_TRAIN

# CNN
model2.add( Conv1D(CNN_FILTERS_L1, 1, activation='relu', padding='same') )
#model.add( MaxPool1D(10) )
model2.add( Conv1D(CNN_FILTERS_L1, 1, activation='relu', padding='same') )
model2.add( GlobalMaxPool1D())

# Dense
model2.add( Dense(20, activation='softmax') )

model2.summary()

#%%
#
# Compile and train
#
from keras.callbacks import ModelCheckpoint

model2.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['acc']
              )

checkpointer2 = ModelCheckpoint(filepath='model2.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

history = model2.fit(x_train, y_train,
                    epochs= 10,
                    batch_size = 128,
                    validation_split = 0.2,
                    callbacks = [checkpointer2]
                    )
            