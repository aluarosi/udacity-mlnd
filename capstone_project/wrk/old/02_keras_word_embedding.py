#%%
from keras.datasets import imdb
from keras import preprocessing

max_features = 10000
maxlen = 20
embedding_dimemsions = 8

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen)

#%%

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(max_features, embedding_dimemsions, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2
                    )

#%%
# Load IMDB directly 

import os

imdb_dir = '/Users/alvaro/Downloads/imdb/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            with open(os.path.join(dir_name, fname)) as f:
                texts.append(f.read())
            labels.append( label_type == 'pos' and 1 or 0 )
                
#%%
# Tokenize 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

#%%
# Data as padded sequences

print(f'Found {len(word_index)} unique tokens.')

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
              
print(f'Shape of data tensor: {data.shape}')
print(f'Shape of labels tensor: {labels.shape}')

# Shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples+validation_samples]
y_val = labels[training_samples:training_samples+validation_samples]

x_test = data[training_samples+validation_samples:]
y_test = labels[training_samples+validation_samples:]

#%%
# Glove embeddings

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
# Embeddings matrix

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, idx in word_index.items():
    if idx < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

#%%
# Model
            
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

model = Sequential()

model.add( Embedding(max_words, embedding_dim, input_length=maxlen) )

model.add( Flatten() )

model.add( Dense(32, activation='relu'))
model.add( Dense(1, activation='sigmoid'))
model.summary()

#%%
# Load embeddings in the model

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

#%%
# Compile and train model

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc']
              )

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val)
                    )
model.save_weights('pre_trained_glove_model.h5')

#%%
# Plot train and validation losses

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

#%%
# Same model w/o pretrained word embeddings

model2 = Sequential()
model2.add( Embedding(max_words, embedding_dim, input_length=maxlen) )
model2.add( Flatten() )
model2.add( Dense(32, activation='relu') )
model2.add( Dense(1, activation='sigmoid') )

model2.summary()

model2.compile(optimizer='rmsprop',
               loss='binary_crossentropy',
               metrics=['acc']
               )

history2 = model2.fit(x_train, y_train,
                     epochs=10,
                     batch_size=32,
                     validation_data = (x_val, y_val)
                     )
#%%
model2_learned_embeddings = model2.layers[0].get_weights()

#%%
# Evaluate model on the test data

test_dir = os.path.join(imdb_dir, 'test')

labels_test = []
texts_test = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            with open(os.path.join(dir_name, fname)) as f:
                labels_test.append(label_type == 'pos' and 1 or 0)
                texts_test.append(f.read())

#%%
sequences_test = tokenizer.texts_to_sequences(texts_test)

x_test = pad_sequences(sequences_test, maxlen=maxlen)
y_test = np.asarray(labels_test)

#%%

model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

