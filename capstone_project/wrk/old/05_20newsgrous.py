#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:39:47 2018

@author: alvaro
"""
"""
This aims to be definitive draft before capstone proposal.
"""

#%%############################################################################
#%%############################################################################
"""
There will be 5 models.

(A) Text sequences -> Embeddings -> CNN layer (2-10 filter length) -> GlobalMaxPool -> Dense 20 softmax (with dropout + l2 constraint)
(B) Text sequences -> Embeddings -> CNN layer (1 filter length) -> GlobalMaxPool -> Dense (ídem)
(C) Text sequences -> Embeddings -> Average -> Dense (ídem))
(D) Text sequences -> Embeddings -> Average -> SVM (RBF/linear?)
(E) Text sequences -> TFIDF -> SVM
"""

#%%############################################################################

###############################################################################
# DEFINITIONS
###############################################################################

#
# Two environments: Google Colab and local (Spyder).
# Set ENV reference.
#
#ENV = 'colab'
ENV = 'local'

#
# Model static configurations
#
NUM_WORDS = 10000 # Max number of words to be included when tokenizing (most frequent words)
EMBEDDING_DIMS = 100 # Number of dimensions for word embedding vectors
MAX_TEXT_LENGTH = 1000 # Max length of a text
N_CATEGORIES = 20 # Number of categories of the problem

###############################################################################
# COLAB configuration
###############################################################################

if ENV == 'colab':
    #Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    #Check GPU
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    
    #Install ggplot (not by default in Colab)
    !pip install ggplot

###############################################################################
# IMPORTS
###############################################################################

import os
from math import log
from functools import reduce
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, Dropout
from keras.constraints import max_norm
from keras.callbacks import ModelCheckpoint
#from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
import ggplot as gg



###############################################################################
# ENVIRONMENT
###############################################################################

BASE_DIR = '/content/drive/My Drive/UDACITY MLND' \
            if ENV == 'colab' else\
            '/Volumes/PENDRIVENEW/live!/PROJECTS/UDACITY_ML_NANODEGREE/my_MLND_projects/nlp-cnn'
WRK_DIR = os.path.join(BASE_DIR, 'wrk')



#%%############################################################################

#
# Import 20newsgroups dataset
#
raw_dataset_train = fetch_20newsgroups(subset='train',
                                    random_state=42)
raw_dataset_test = fetch_20newsgroups(subset='test',
                                   random_state=42)

dataset_train = fetch_20newsgroups(subset='train',
                                    random_state=42,
                                    remove=('headers', 'footers', 'quotes'))
dataset_test = fetch_20newsgroups(subset='test',
                                   random_state=42,
                                   remove=('headers', 'footers', 'quotes'))

#%%############################################################################

texts_train = dataset_train.data
texts_test = dataset_test.data

targets_train = dataset_train.target
targets_test = dataset_test.target

target_names = dataset_train.target_names

def get_target_name(target_number):
    try:
        return target_names[target_number]
    except:
        return None

#%%############################################################################

#
# Analysis of dataset
#

print()
print('TARGET NAMES:')
print('\n'.join(target_names))
print()

tr_N = len(texts_train)
te_N = len(texts_test)
N = tr_N + te_N

print(f"Training dataset: {tr_N} documents.")
print(f"Test dataset: {te_N} documents.")
print(f"Total in 20 Newsgroup dataset: {N} documents.")

#%%############################################################################



# Split per class (train dataset)

named_target = map(lambda x: target_names[x]  , targets_train)

mydf = pd.DataFrame({'target number': dataset_train.target, 'target name': list(named_target) }  )

classes_summary = mydf.groupby('target name', as_index=False).count().rename(columns={'target number':'count'})

sns.set(style="whitegrid", rc={'figure.figsize':(11.7,8.27)})
ax = sns.barplot(y="target name", x="count", color="grey", data=classes_summary)
ax.set_title("20 Newsgrous - Number of documents by class")

#%%############################################################################


# Entropy calculation


def H(l):
    '''Receives a list of integers that represent number of occurrences of each class.
    '''
    total = sum(l)

    probabilities = map(lambda x: x/total  , l)

    def plog2(x):
        if x == 0.0:
            return 0
        else:
            return (- x * log(x,2))

    informations = map(plog2 , probabilities)
    return sum(informations)

#%%############################################################################

# Entropy of classes

entropy = H(classes_summary['count'].values)

print(f'Entropy of training set: {entropy:.3f} bits.')

#%%############################################################################

# Document sample

def print_raw_and_clean_document(idx):
    print('========================================')
    print(texts_train[idx])
    print('========================================')
    print(raw_dataset_train.data[idx])
    print('========================================')

print_raw_and_clean_document(201)
print_raw_and_clean_document(302)
print_raw_and_clean_document(403)

#%%############################################################################

# Text distribution (clean texts, without headers, footers, quotes)

text_lengths = pd.DataFrame({
        'length': [len(t) for t in texts_train] ,
        'target name': [get_target_name(t) for t in targets_train] })

print(text_lengths.describe())

#%%############################################################################

sns.set(style="whitegrid", rc={'figure.figsize':(16.0,10.0)})
ax = sns.distplot(text_lengths['length'], hist=False, rug=True);

ax.set_title("20 Newsgrous - Doucment length distribution")

#%%############################################################################

sns.set(style="whitegrid", rc={'figure.figsize':(16.0,2.0)})
ax = sns.boxplot(x='length', data=text_lengths);

ax.set_title("20 Newsgrous - Doucment length distribution")

#%%############################################################################

sns.set(style="whitegrid", rc={'figure.figsize':(16.0,10.0)})
ax = sns.boxplot(x="length", y="target name", color="grey", data=text_lengths)
ax.set_title("20 Newsgrous - Length of documents by class")


#%%############################################################################

###############################################################################
# Preprocessing text to bag of words and text sequences
###############################################################################

texts_train_truncated = [ t[:MAX_TEXT_LENGTH] for t in texts_train ]
texts_test_truncated = [ t[:MAX_TEXT_LENGTH] for t in texts_test ]

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(texts_train_truncated)

def inverted_word_index(idx):
    """ Gets the word for a numeric index.
        It's the inverted index of tokenizer.word_index
    """
    words, idxs = zip(*tokenizer.word_index.items())
    inverted_word_index = dict(zip(idxs, words))
    return inverted_word_index.get(idx)

#%%############################################################################


#
# Input matrices - TRAIN
#

X_bagofwords_train = tokenizer.texts_to_matrix(texts_train_truncated, mode = 'binary')
X_count_train = tokenizer.texts_to_matrix(texts_train_truncated, mode='count')
X_tfidf_train = tokenizer.texts_to_matrix(texts_train_truncated, mode='tfidf')
X_seqs_train = pad_sequences(tokenizer.texts_to_sequences(texts_train), maxlen=MAX_TEXT_LENGTH)

#
# Input matrices - TEST
#
X_tfidf_test = tokenizer.texts_to_matrix(texts_test_truncated, mode='tfidf')
X_seqs_test = pad_sequences(tokenizer.texts_to_sequences(texts_test), maxlen=MAX_TEXT_LENGTH)

#
# Output matrices - TRAIN
#
Y_train = targets_train
Y_1hot_train = to_categorical(np.asarray(Y_train), N_CATEGORIES)

#
# Output matrices - TEST
#
Y_test = targets_test
Y_1hot_test = to_categorical(np.asarray(Y_test), N_CATEGORIES)

#%%############################################################################

###############################################################################
#
# Load Glove embeddings - 100 dimensions
#
###############################################################################

# After F. Chollet's book - Chapter 6

glove_dir = WRK_DIR

embeddings_index = {}

with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        line_as_list = line.split()
        token = line_as_list[0]
        coords = np.asarray( line_as_list[1:], dtype='float32' )
        embeddings_index[token] = coords

print( f'Found {len(embeddings_index)} vectors.' )

#
# Embeddings matrix
#

# After F. Chollet's book - Chapter 6

EMBEDDING_MATRIX = np.zeros((NUM_WORDS, EMBEDDING_DIMS))

for word, idx in tokenizer.word_index.items():
    if idx < NUM_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            EMBEDDING_MATRIX[idx] = embedding_vector


#%%############################################################################

###############################################################################
#
# (A, B) Text sequences -> Embeddings -> CNN (size 1-10) -> Global Max Pool ->  Dense 20
#
###############################################################################

def create_keras_model(**kw):
  
    embedding_preload = not not kw.get('embedding_preload') # Defaults to False
    embedding_train = not not kw.get('embedding_train') # Defaults to False
    embedding_matrix = EMBEDDING_MATRIX if embedding_preload \
                       else np.zeros((NUM_WORDS, EMBEDDING_DIMS))

    cnn_num_filters = kw.get('cnn_num_filters') or 100
    cnn_filter_size = kw.get('cnn_filter_size') or 2

    dropout = kw.get('dropout') or 0.5
    max_norm_constraint = kw.get('max_norm_constraint') or 3.0

    optimizer = kw.get('optimizer') or 'rmsprop'

    print(f'''Creating model with params:
          embedding_preload={embedding_preload},
          embedding_train={embedding_train},
          cnn_num_filters={cnn_num_filters},
          cnn_filter_size={cnn_filter_size},
          dropout={dropout},
          max_norm_constraint={max_norm_constraint},
          optimizer={optimizer},
          embedding_matrix={embedding_matrix}
          ''')

    model = Sequential()
    model.add( Embedding(NUM_WORDS, EMBEDDING_DIMS, input_length = MAX_TEXT_LENGTH) )

    # Preload embeddings (Glove)
    if embedding_preload:
        model.layers[0].set_weights( [embedding_matrix] )

    # Do we have to train embedding layer?
    model.layers[0].trainable = embedding_train

    # CNN layer
    model.add( Conv1D(cnn_num_filters, cnn_filter_size, activation='relu', padding='same') )
    model.add( GlobalMaxPool1D())

    # DENSE layer - softmax
    # Dense
    model.add( Dropout(dropout) )
    model.add( Dense(N_CATEGORIES,
                     activation='softmax',
                     kernel_constraint=max_norm(max_norm_constraint)) )

    # TODO: change optimizer from RMSPROP to SGD as per paper?
    model.compile(optimizer=optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['acc']
                  )

    return model

#%%############################################################################

#
# Search grid, training sessions and reporting
#

def calculate_params_from_grid(grid):

    keys = list(grid.keys())
    values = list(grid.values())

    def cartesian(A,B):

        result = []

        for a in A:
            for b in B:
                # Check if not tuple nor list, then wrap them as tuples
                if type(a) != tuple and type(a) != list:
                    a = (a,)
                if type(b) != tuple and type(b) != list:
                    b = (b,)

                result.append(a+b)

        return result

    result = [dict(zip(keys,t))  for t in reduce(cartesian, values)]
    print( f'Grid of {len(result)} hyperparameter combinations.' )
    return result

def fit_keras_model(model, params, X, Y, idx=0, session='undefined',
                    save_weights=True,
                    weights_filepath=os.path.join(WRK_DIR,'weights.hdf5')):

    validation_split = 0.2

    print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
    print( datetime.now() )
    print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )

    print(f'Fitting model {idx} with params:')
    print(params)
    print()

    # Checkpointer for saving best weights
    callbacks = []

    if save_weights == True:
        checkpointer = ModelCheckpoint(weights_filepath,
                                       monitor='val_acc',
                                       verbose=1, 
                                       save_best_only=True)
        callbacks.append(checkpointer)

    time_start = datetime.now()
    history = model.fit(X,Y, epochs = params.get('epochs'),
                        batch_size = params.get('batch_size'),
                        validation_split = validation_split,
                        callbacks = callbacks)
    time_end = datetime.now()


    val_acc_sequence = history.history['val_acc']
    best_epoch = val_acc_sequence.index(max(val_acc_sequence))
    
    print()

    result = {'__type__' : 'keras',
              'session' : session,
              'history' : history.history,
              'params' : params,
              'idx' : idx,
              'training_time' : time_end - time_start,
              'starting_time' : time_start,
              'weights_file' : weights_filepath,
              'best_epoch' : best_epoch,
              'best_score' : val_acc_sequences[best_epoch]
              }
    return result

def run_keras_training_session(params, session_id = None, weights_filepath=None):
    ''' Returns sequence of reports'''

    n = datetime.now()
    session_id = session_id or str(n.year)+str(n.month)+str(n.day)+str(n.hour)+str(n.minute)+str(n.second)

    models = [create_keras_model(**p) for p in params]

    reports = [fit_keras_model(m, p, 
                               X_seqs_train, Y_1hot_train, 
                               idx+1, session_id,
                               weights_filepath=weights_filepath) \
               for (p, m, idx) in zip(params, models, range(len(params)))]

    return (reports, models)

def run_keras_training_session_from_grid(grid, session_id=None):

    params = calculate_params_from_grid(grid)
    return run_keras_training_session(params, session_id)

#
# Save and retrieve reports and model
#

def save_reports(reports, filepath):
    with open(filepath,'wb') as f:
        pickle.dump(reports, f)

def load_reports(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def save_model(model, filepath='/tmp/model.json'):
    ''' Saves model description (.json)'''
    with open(filepath, 'w') as f:
        f.write(model.to_json())
    return filepath


def load_model(filepath='/tmp/model.json'):
    '''Loads model from json in file'''
    with open(filepath, 'r') as f:
        return model_from_json(f.read())
    
#
# Sort models by validation score
#

def sort_models_by_score(reports, score_metric='val_acc'):

    matrix = np.array([r['history'][score_metric] for r in reports])

    array_of_maxs = matrix.max(axis=-1)


    aslist = [ (i,v) for (i, v) in enumerate(list(array_of_maxs)) ]
    sorted_indices = [ i for (i,v) in sorted(aslist, key = lambda x: x[1], reverse=True) ]
    sorted_reports = [ reports[i] for i in sorted_indices ]

    # Enrich with best score and best epoch (mutates!)
    for report in sorted_reports:
        l = report['history'][score_metric]
        m = max(l)
        report['best_score'] = m
        report['best_epoch'] = l.index(m)

    return sorted_reports

#%%############################################################################

#
# Plot
#

def plot_epochs(report, renderer='ggplot'):

    history = report['history']
    params = report['params']

    accuracies = history['acc']
    accuracies_val = history['val_acc']
    indexes = range(len(accuracies))

    if renderer == 'pyplot':

        plt.plot(indexes, accuracies, 'bo', label='Training')
        plt.plot(indexes, accuracies_val, 'b', label='Validation')
        plt.title(f'Accuracy - Epoch : {params}')
        plt.legend()
        return plt

    elif renderer == 'ggplot':

        data = pd.DataFrame(
                {"train": accuracies,
                 "validation": accuracies_val,
                 "epoch": range(len(accuracies)),
                 })
        data_melted = data.melt(id_vars="epoch")

        p = gg.ggplot(data_melted, gg.aes(x="epoch", y="value", color="variable"))
        p = p + gg.geom_point() + gg.geom_line()
        p = p + gg.ggtitle(f'Accuracy - Epoch : {params}')
        return p

    else:
        raise(Exception(f'Renderer {renderer} not supported.'))


#%%############################################################################

#
# Previous grid search for finding best non-relevant hyperparameters
# This is grid is for exploring optimizer params
# and embedding boolean params.
#

grid0 = {'embedding_preload': [True, False],
         'embedding_train': [True, False],
         'cnn_filter_size': [4],
         'cnn_num_filters': [300],
         'batch_size': [64,128,256,512],
         'epochs': [40],
         'optimizer': ['rmsprop', 'sgd','adagrad', 'adadelta', 'adam', 'adamax', 'nadam' ]
         }


#%%############################################################################

reports0, models0 = run_keras_training_session_from_grid(grid0)


#%%############################################################################

best_report0 = sort_models_by_score(reports0)[0]


#%%############################################################################

# Rank of reports

best_reports0_topN = sort_models_by_score(reports0)[0:]

#%%############################################################################

#
# Trained in Collab 2018-11-17 - Copy/paste
import datetime

best_reports0_topN = [{'__type__': 'keras',
  'best_epoch': 14,
  'best_score': 0.6977463523950715,
  'history': {'acc': [0.06993702348885965,
    0.1609766874201007,
    0.2965418181950421,
    0.4322174348339641,
    0.5391669422141123,
    0.6169484041498691,
    0.6692078232620319,
    0.7147276547913682,
    0.7528449902085622,
    0.776599270864661,
    0.8080875051112455,
    0.8251021982898845,
    0.8388023420257893,
    0.8544912163940744,
    0.8707325149011828,
    0.8753728860781979,
    0.8853165405703545,
    0.8945972821275485,
    0.8987957132928162,
    0.9014473547923368],
   'loss': [2.983871232763916,
    2.9072206707346337,
    2.7234965161241464,
    2.3688095614549507,
    1.9698941113474409,
    1.654686217676301,
    1.4118254398617371,
    1.2232825719458058,
    1.0623702882284776,
    0.9281553893560284,
    0.8209499342772674,
    0.7442747206072059,
    0.6783263579796275,
    0.6135825214168931,
    0.5555561954323698,
    0.5226056160761919,
    0.4880347003391231,
    0.457978564192611,
    0.43707428345257715,
    0.41035438648456635],
   'val_acc': [0.11268228047843849,
    0.2894387986746395,
    0.41670349045396116,
    0.5103844445572455,
    0.5793194893288286,
    0.618205923608484,
    0.657092353647597,
    0.6699072004839535,
    0.6774193567351013,
    0.6889085279225355,
    0.6871409613212758,
    0.6849315087457067,
    0.6942112227482865,
    0.6924436568581737,
    0.6977463523950715,
    0.695536897686062,
    0.6933274426346483,
    0.6920017655634322,
    0.6924436575693206,
    0.6946531126207343],
   'val_loss': [2.9643828501176688,
    2.866641445492834,
    2.586612553813528,
    2.164858698476167,
    1.819259567751697,
    1.5847095700422853,
    1.4301645514504655,
    1.3254467567089012,
    1.251233743030415,
    1.1951860386451894,
    1.1574699239187203,
    1.1321830044686136,
    1.1085478015247299,
    1.1007030842528636,
    1.0913883839041272,
    1.0882059539570725,
    1.0860509904569828,
    1.0867509656493979,
    1.079435949666912,
    1.0811083029710347]},
  'idx': 28,
  'params': {'batch_size': 512,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'nadam'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 15, 20, 373052),
  'training_time': datetime.timedelta(0, 50, 652234),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 14,
  'best_score': 0.6911179872144917,
  'history': {'acc': [0.0778919456635914,
    0.2706883217422834,
    0.45088940450215576,
    0.5535299967480106,
    0.6328582479049982,
    0.6909733732594086,
    0.7438957022772403,
    0.7754944206443153,
    0.8080875042551408,
    0.8280852945150426,
    0.845431444138114,
    0.8618937134808591,
    0.8736051265777982,
    0.879129378028554,
    0.8950392222512481,
    0.9006739587642095,
    0.9038780245869453,
    0.9053143299675662,
    0.9160313777746568,
    0.913269252082206],
   'loss': [2.9724649782668635,
    2.7323941684746504,
    2.207478541443327,
    1.788317233459052,
    1.4593051372608576,
    1.2164972486080827,
    1.0224974050783686,
    0.892597825043031,
    0.7678500354494157,
    0.694802030919336,
    0.628667764624974,
    0.5570266838759141,
    0.5118501940335691,
    0.47506648465067414,
    0.4360524198255833,
    0.40656876593660274,
    0.3861727923790324,
    0.38966269652009233,
    0.34786758391504957,
    0.3465149521287693],
   'val_acc': [0.23464427778429023,
    0.4441007521238888,
    0.5373398156435786,
    0.6089262053653618,
    0.634997792097514,
    0.6597437038918907,
    0.6641626186040035,
    0.6619531596544512,
    0.6765355709586268,
    0.6853733971958607,
    0.6866990710800852,
    0.6836058302521969,
    0.6866990696577914,
    0.6840477233116362,
    0.6911179872144917,
    0.6889085300296374,
    0.6805125943759981,
    0.6844896160286715,
    0.6862571790741968,
    0.684047725076334],
   'val_loss': [2.915659282779567,
    2.4650581231751487,
    1.930901768299209,
    1.6128128355601905,
    1.4153190486156144,
    1.297833591449382,
    1.2281590340361888,
    1.1918377742480768,
    1.1558761409674394,
    1.1377881101628504,
    1.1288239504982294,
    1.1281242682983323,
    1.1171579605385733,
    1.1241148047040534,
    1.1139219221265981,
    1.1304103621222832,
    1.1358409265053784,
    1.145618122233999,
    1.1425453084683554,
    1.1409745957542718]},
  'idx': 21,
  'params': {'batch_size': 256,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'nadam'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 9, 28, 84640),
  'training_time': datetime.timedelta(0, 53, 106124),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 15,
  'best_score': 0.689350420850281,
  'history': {'acc': [0.09490664020788207,
    0.34935366258859873,
    0.5204949731945847,
    0.6349574634415341,
    0.6942879241904479,
    0.7484255884842456,
    0.7909623247103827,
    0.8248812287449903,
    0.8491879351730967,
    0.8614517733769158,
    0.8765882223880748,
    0.8858689648343007,
    0.8928295216656774,
    0.8986852282042689,
    0.9090708209235238,
    0.9096232461686978,
    0.9134902221144213,
    0.9148160424789347,
    0.9194564137218039,
    0.9225499945086756],
   'loss': [2.9492045858396034,
    2.4383558423568394,
    1.782463447624327,
    1.3839948867358052,
    1.1486471868090808,
    0.9442239046794857,
    0.7963143777167675,
    0.6770879054527864,
    0.5936282924980949,
    0.5356798540484298,
    0.48914379054651036,
    0.44880295187027436,
    0.4169922133016792,
    0.39177588421907783,
    0.36359272571594436,
    0.34593179424537923,
    0.3354304455769759,
    0.32298824363960743,
    0.31567067545351435,
    0.30209347538204695],
   'val_acc': [0.3084401235451911,
    0.5059655313991205,
    0.6106937696751482,
    0.6349977908859303,
    0.660627484690337,
    0.6694653125869138,
    0.6752098994185531,
    0.6805125933751247,
    0.6871409630859736,
    0.6875828556713149,
    0.6822801601344171,
    0.6853733991976075,
    0.6827220504282852,
    0.6889085282649395,
    0.6862571804964905,
    0.689350420850281,
    0.6889085285546661,
    0.6862571817870905,
    0.6849315069019927,
    0.6836058320168947],
   'val_loss': [2.811581563802088,
    1.9667673510257722,
    1.5306825330180507,
    1.33833412111622,
    1.240335605758721,
    1.1734970588363802,
    1.1421289736330114,
    1.127540950642399,
    1.1204411553114428,
    1.11892113077709,
    1.1181214094899472,
    1.126560537308336,
    1.1281828736500354,
    1.1288090342712571,
    1.1274870060947264,
    1.1356481728804517,
    1.1460245382032508,
    1.1484601445499374,
    1.1539222067876354,
    1.1610756834892377]},
  'idx': 14,
  'params': {'batch_size': 128,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'nadam'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 3, 20, 255252),
  'training_time': datetime.timedelta(0, 56, 496872),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 17,
  'best_score': 0.6875828556713149,
  'history': {'acc': [0.06651198764790252,
    0.18705115465745914,
    0.34769638725478746,
    0.48293006327281063,
    0.573417302205831,
    0.6297646671181265,
    0.6760578943990025,
    0.7300850736832779,
    0.7560490554847081,
    0.786763893637311,
    0.8099657497397049,
    0.819798917338909,
    0.8444370789009489,
    0.8510661805918066,
    0.8702905757389547,
    0.8757043421340903,
    0.8790188930190317,
    0.8859794498569938,
    0.8994586234026332,
    0.8989061983023384],
   'loss': [2.987558285026521,
    2.881855010222419,
    2.5918117704318777,
    2.1435727976446137,
    1.7749878961254175,
    1.5144036396104794,
    1.313020884048977,
    1.1371401598566528,
    1.0097696524392885,
    0.9041637920527258,
    0.8065207039976684,
    0.739730181970355,
    0.6652156017377434,
    0.6182831411150435,
    0.5676443253863702,
    0.5300956630743665,
    0.5026204173123161,
    0.4739429127470642,
    0.4310162994278398,
    0.4230832308843615],
   'val_acc': [0.17543084413748214,
    0.36279275307495196,
    0.46619531632101857,
    0.5519222273428357,
    0.5868316399171402,
    0.6235086165115046,
    0.6456031806691263,
    0.6535572249753465,
    0.6650463996394989,
    0.6725585506492308,
    0.6787450297764851,
    0.6778612461861288,
    0.6822801601344171,
    0.6787450297764851,
    0.684489616318398,
    0.6813963775449341,
    0.6836058340186415,
    0.6875828556713149,
    0.6871409643765735,
    0.6836058320168947],
   'val_loss': [2.9609200522481367,
    2.8015079263349154,
    2.363453325211765,
    1.9239126798382808,
    1.656198584448696,
    1.483516515223157,
    1.3692574643793751,
    1.2899741717687077,
    1.2377072569020702,
    1.1986216692286047,
    1.1720580401485914,
    1.1512256311306215,
    1.137239980950484,
    1.1248995722051327,
    1.1150459740681296,
    1.11036317147252,
    1.1041519738802907,
    1.1019316706254954,
    1.1001816161480615,
    1.1041133097721527]},
  'idx': 12,
  'params': {'batch_size': 128,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adam'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 1, 28, 813202),
  'training_time': datetime.timedelta(0, 56, 5162),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 16,
  'best_score': 0.6858152895704919,
  'history': {'acc': [0.08297425699064763,
    0.2903546569769112,
    0.4673516738613644,
    0.5698817809857448,
    0.6411446249164965,
    0.7014694508169649,
    0.7487570434601291,
    0.7770412109356771,
    0.8181416417811934,
    0.8290796597258661,
    0.8494089051328724,
    0.8664235996022539,
    0.8745994918347195,
    0.8873052702215071,
    0.891945641319497,
    0.9023312341836313,
    0.9038780245013348,
    0.9111700364139618,
    0.9115014916137495,
    0.9153684676582544],
   'loss': [2.9715681130840506,
    2.6706482353901655,
    2.1222176135356827,
    1.6979280976929885,
    1.414267273847997,
    1.2037460509617282,
    1.005287667234067,
    0.8758303357637887,
    0.7613378205545409,
    0.6854825899121827,
    0.6126904415565358,
    0.5473251865030244,
    0.5001762853878767,
    0.46424489265093555,
    0.44360976764252535,
    0.40587301403629805,
    0.3876018792394116,
    0.36252283427336246,
    0.3548148910573353,
    0.3461586126186955],
   'val_acc': [0.26071586387114243,
    0.44984533799416293,
    0.53778170543701,
    0.5996464867008191,
    0.6323464425116896,
    0.6460450726486759,
    0.6650463988229968,
    0.6663720727072213,
    0.6725585507282471,
    0.677861246265145,
    0.6743261159072131,
    0.680512594033594,
    0.6813963767284321,
    0.676977463675662,
    0.6844896157916225,
    0.683163941907398,
    0.6858152895704919,
    0.6809544854336906,
    0.6805125939282389,
    0.6800707027388525],
   'val_loss': [2.90330519562425,
    2.3483461281181803,
    1.8507483499639075,
    1.5702112393731098,
    1.4078921877173687,
    1.3024196815975355,
    1.2236280195280034,
    1.1851311322602875,
    1.1588266772397893,
    1.1390675990074755,
    1.122607631384141,
    1.1164099188226015,
    1.1115419484322648,
    1.1052416076820295,
    1.1042772242309664,
    1.1032953742221558,
    1.1102856780594552,
    1.1115574938398622,
    1.1173337337668365,
    1.1273910054789673]},
  'idx': 5,
  'params': {'batch_size': 64,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adam'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 54, 38, 50388),
  'training_time': datetime.timedelta(0, 64, 268460),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 18,
  'best_score': 0.6853733989605586,
  'history': {'acc': [0.06286598166401261,
    0.13357640042244143,
    0.2329024417240861,
    0.3457076566816759,
    0.4477958236823568,
    0.5250248594015899,
    0.5939675176516381,
    0.6402607448403183,
    0.6867749421863368,
    0.7150591096750558,
    0.7404706663696058,
    0.7720693847498517,
    0.7938349354716246,
    0.8104076898897462,
    0.8234449233248569,
    0.8379184621471738,
    0.8446580489463351,
    0.8638824440144581,
    0.8625566236894574,
    0.8759253122058182],
   'loss': [2.991329524335565,
    2.9359341285173675,
    2.843665643550193,
    2.653727030914774,
    2.370961281307862,
    2.061608913115751,
    1.7938394991551534,
    1.5753415586039343,
    1.3960723925026697,
    1.2485140816781577,
    1.1211440727574011,
    1.0120653960602335,
    0.9256930115828527,
    0.8389980387073922,
    0.7783792100352037,
    0.7079284132784762,
    0.6678732815216607,
    0.6070597001484462,
    0.5843620750406489,
    0.5437882674450295],
   'val_acc': [0.12903225766284984,
    0.24524966989846764,
    0.38223597053084496,
    0.45382236096377504,
    0.49403446665181033,
    0.5497127711851935,
    0.5974370324132298,
    0.6204153783174937,
    0.6389748148037383,
    0.653557224343216,
    0.6593018097525617,
    0.6650464008247436,
    0.6716747663477275,
    0.6827220480051179,
    0.6787450263524445,
    0.6765355737768756,
    0.680954482826152,
    0.6805125922425574,
    0.6853733989605586,
    0.6853733989605586],
   'val_loss': [2.9743617714583954,
    2.9289309763560714,
    2.8059631128306943,
    2.547545489936185,
    2.2222043486016543,
    1.9372347053362635,
    1.728416658480295,
    1.572065987999967,
    1.4554577887822082,
    1.37155155265326,
    1.3059481902297387,
    1.2566336110477894,
    1.2210158173931525,
    1.1893886013785349,
    1.1630196258445467,
    1.147323028945417,
    1.133374745636138,
    1.1216758164061522,
    1.1118590165912394,
    1.1058492399241826]},
  'idx': 19,
  'params': {'batch_size': 256,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adam'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 7, 42, 535292),
  'training_time': datetime.timedelta(0, 52, 962557),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6840477261562238,
  'history': {'acc': [0.05159650861339442,
    0.10253010722398495,
    0.15335322077525096,
    0.24704452531862728,
    0.32493647108840856,
    0.382278201682385,
    0.4544249255312647,
    0.5104408361953657,
    0.5601590994233888,
    0.6036901998922926,
    0.6388244396243329,
    0.6747320748840085,
    0.699149265129676,
    0.7243398510083455,
    0.7468787980941024,
    0.7657717371458455,
    0.7812396430492454,
    0.795160755197352,
    0.8109601159778541,
    0.8185835829980728],
   'loss': [2.9994741328288645,
    2.9591476925696183,
    2.922286173610078,
    2.8642786676324463,
    2.771339467237989,
    2.6318129382598676,
    2.425325592688178,
    2.1965696623596402,
    1.9774150569035116,
    1.8010915046554286,
    1.6383422052197767,
    1.4855766789766454,
    1.372193122750395,
    1.2592378310400694,
    1.1551200411304192,
    1.0699074407556863,
    1.0001295393652025,
    0.9161418703376568,
    0.8704247394591692,
    0.8179250473067874],
   'val_acc': [0.0786566504425856,
    0.16040653993848464,
    0.2642509941698122,
    0.3645603169764874,
    0.40565620737427693,
    0.4467520988256174,
    0.4825452929885315,
    0.5130357937348189,
    0.5461776390625637,
    0.5691559867315255,
    0.5965532466367552,
    0.6142289019558106,
    0.6296950933736244,
    0.6429518364564117,
    0.6495802009258447,
    0.6593018111748554,
    0.6623950509491927,
    0.6699072029597981,
    0.6783031361375927,
    0.6840477261562238],
   'val_loss': [2.984036383352815,
    2.9667156991103134,
    2.936799911814246,
    2.8810526913791645,
    2.778736075415213,
    2.606992111235554,
    2.379854280443988,
    2.1467560370095895,
    1.94846468297349,
    1.7902271030647692,
    1.6664103422341914,
    1.5636545117323872,
    1.48129272192282,
    1.412981762961903,
    1.3542309732917552,
    1.3081441280644814,
    1.270623167426092,
    1.2385013961602354,
    1.2116963131154894,
    1.1892681126040376]},
  'idx': 26,
  'params': {'batch_size': 512,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adam'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 13, 39, 838584),
  'training_time': datetime.timedelta(0, 50, 392250),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6840477248919626,
  'history': {'acc': [0.07148381397202191,
    0.20704894500297152,
    0.3347696388160283,
    0.4263617280352486,
    0.4883438294209929,
    0.54237100873161,
    0.5945199428177871,
    0.6298751521210633,
    0.6597061100065109,
    0.6890951278341131,
    0.7148381395901571,
    0.7396972711975832,
    0.760689426707821,
    0.7800243068315643,
    0.7953817259012803,
    0.8120649653091679,
    0.8207932825760741,
    0.830626450221376,
    0.8415644681199508,
    0.8507347255566544],
   'loss': [2.9821120388217293,
    2.8629768189244897,
    2.6174197985103627,
    2.309912945944727,
    2.046841515663407,
    1.834717120495755,
    1.6527729049570297,
    1.495385712898182,
    1.3803046464841018,
    1.262224320324333,
    1.1723824886837897,
    1.0705828932738017,
    0.9880417854189807,
    0.9241383670772746,
    0.8572604878411585,
    0.7973196585196024,
    0.7464557330050714,
    0.702893258247464,
    0.6647611745562083,
    0.6330907340866863],
   'val_acc': [0.18824569153353748,
    0.3654441011331274,
    0.4334953600887277,
    0.4697304458229404,
    0.519664162260423,
    0.5558992492852356,
    0.5775519224378424,
    0.6080424211955524,
    0.6186478125590748,
    0.6261599645696801,
    0.632788333701076,
    0.6456031822494527,
    0.6517896603758336,
    0.663278832458786,
    0.6703490934643764,
    0.6668139643970444,
    0.6791869203600797,
    0.6809544855390457,
    0.6822801594232701,
    0.6840477248919626],
   'val_loss': [2.948849087945666,
    2.7751324192591595,
    2.460154773385136,
    2.1509453117768427,
    1.9187396756853616,
    1.7476523921282165,
    1.6172391301069542,
    1.5132049207291243,
    1.4369977820093423,
    1.3765005951058376,
    1.332879362409746,
    1.296885404131658,
    1.2638069451303013,
    1.2399559704223881,
    1.2194988715453745,
    1.2008371882558764,
    1.1877290175332993,
    1.1791354105423049,
    1.1744024047691424,
    1.1630802591237883]},
  'idx': 8,
  'params': {'batch_size': 128,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'rmsprop'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 57, 52, 289006),
  'training_time': datetime.timedelta(0, 54, 170194),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6774193549704035,
  'history': {'acc': [0.08010164622775942,
    0.2446138547864517,
    0.3786321953474099,
    0.46757264393967773,
    0.5291128052280643,
    0.580709313947237,
    0.6241299304076025,
    0.6589327146566819,
    0.6890951276233797,
    0.7240083967897856,
    0.7473207380992644,
    0.7587006960490991,
    0.7878687437127775,
    0.8021213125094645,
    0.8174787316713764,
    0.8251021987376931,
    0.8391337973441146,
    0.8476411446446596,
    0.8610098331412641,
    0.860125953018988],
   'loss': [2.971838787897768,
    2.7442511183532505,
    2.3715813633829517,
    2.045514214777733,
    1.7746168895131804,
    1.5829594257378128,
    1.4327972702868879,
    1.2905546244093495,
    1.1957683571266635,
    1.0720868433474071,
    0.9884768511600724,
    0.931242556114142,
    0.8562649997502134,
    0.7928468312970605,
    0.7396221521901447,
    0.7079974669354382,
    0.6524439468105621,
    0.6137588854692692,
    0.5765686169498837,
    0.5535065487674186],
   'val_acc': [0.2399469730775545,
    0.3831197526067219,
    0.4591250551837343,
    0.5117101193633271,
    0.551922227184803,
    0.58506407418506,
    0.6080424214062625,
    0.6133451170485156,
    0.6248342906064392,
    0.6420680511013576,
    0.6420680509960025,
    0.6517896594803153,
    0.6579761376066963,
    0.6601855945808404,
    0.6597437032860989,
    0.6632788331435941,
    0.6610693766698866,
    0.6668139640019628,
    0.6672558552967043,
    0.6774193549704035],
   'val_loss': [2.904805883073828,
    2.52463031963495,
    2.1512376891265546,
    1.8710487249726697,
    1.6673141032049101,
    1.530057363946302,
    1.4322730117385702,
    1.3674911029877201,
    1.3212266146372864,
    1.2831904977704907,
    1.2510505055517314,
    1.2372230083817042,
    1.2172035810118274,
    1.1975081309880296,
    1.189812034671411,
    1.1783774671331213,
    1.1749337715645358,
    1.1679445626827347,
    1.159410806129951,
    1.157797216526239]},
  'idx': 1,
  'params': {'batch_size': 64,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'rmsprop'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 50, 24, 289817),
  'training_time': datetime.timedelta(0, 67, 464126),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 18,
  'best_score': 0.6721166596442159,
  'history': {'acc': [0.11490443048095482,
    0.3408463153802496,
    0.4684565243978103,
    0.5370677273394113,
    0.5839133800004624,
    0.6225831401228259,
    0.6572754393426269,
    0.6773837146383928,
    0.7016904211191825,
    0.7161639599414994,
    0.7316318640339086,
    0.7420174567531636,
    0.7572643908396991,
    0.7626781572282492,
    0.7675394985494317,
    0.7844437080290473,
    0.7920671749899972,
    0.7951607558361379,
    0.8069826539755263,
    0.8137222407549314],
   'loss': [2.929862089544854,
    2.567007976083542,
    2.208325339330387,
    1.9504583022196955,
    1.7725798737316576,
    1.6179258341868799,
    1.4876946318335276,
    1.4009087508903026,
    1.3105649493330949,
    1.2339147853661658,
    1.175691161810555,
    1.1184058109906059,
    1.0742619322513483,
    1.0249135507337166,
    0.9952717378658164,
    0.9416446137554724,
    0.9112986099613317,
    0.8786101686060198,
    0.8532119742441857,
    0.8216343904421377],
   'val_acc': [0.36323464423799956,
    0.47547503336572666,
    0.5289438795948577,
    0.5647370751800656,
    0.5890410961011211,
    0.6115775528440842,
    0.6230667265073631,
    0.633672116290559,
    0.6380910305285739,
    0.6473707477181452,
    0.6500220941959942,
    0.6562085720326486,
    0.6566504633273902,
    0.660185593685322,
    0.662836941453771,
    0.6676977456959274,
    0.6690234205810254,
    0.6707909857599913,
    0.6721166596442159,
    0.6703490944652498],
   'val_loss': [2.784117463095442,
    2.343775360867664,
    2.048922844392454,
    1.8505948990570251,
    1.7136639632116732,
    1.611253777542738,
    1.5319191970711412,
    1.471409865622493,
    1.4239329964093703,
    1.3844678039525495,
    1.3509805840936004,
    1.3218968053123248,
    1.2969412491272048,
    1.2753902312880114,
    1.2581168687612252,
    1.2405686574966255,
    1.2268085602475014,
    1.215988076370583,
    1.2071037886683202,
    1.1980384626169411]},
  'idx': 10,
  'params': {'batch_size': 128,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adagrad'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 59, 39, 597002),
  'training_time': datetime.timedelta(0, 53, 669536),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 16,
  'best_score': 0.672116659038424,
  'history': {'acc': [0.1361175560546888,
    0.4374102309433456,
    0.5874489007529837,
    0.67219091807791,
    0.7413545465445652,
    0.7838912827509462,
    0.8149375760440681,
    0.8437741686594235,
    0.8567009170454992,
    0.8695171803957111,
    0.882996354059888,
    0.8868633299726845,
    0.8981328029590948,
    0.8947077671831688,
    0.9081869406892955,
    0.9091813059528023,
    0.906198210162282,
    0.9120539167140442,
    0.913269252082206,
    0.918241078353642],
   'loss': [2.889261145576458,
    2.092677901863538,
    1.519438615728776,
    1.2038498262505073,
    0.9585968492676732,
    0.7959485481805056,
    0.6776180954193387,
    0.5881934520974632,
    0.529372692509865,
    0.4827344540159632,
    0.43455588746638524,
    0.411198498711851,
    0.3888702057116783,
    0.38350072984484174,
    0.34003808195683694,
    0.32917496809615915,
    0.3349022247023167,
    0.32331714947149115,
    0.3070781359918446,
    0.29491816360164697],
   'val_acc': [0.4030048608569197,
    0.5651789658163376,
    0.6226248341327318,
    0.6486964204171248,
    0.6500220943013494,
    0.6597437028910172,
    0.6584180290067928,
    0.6588599203015343,
    0.6712328765542961,
    0.6659301810173982,
    0.6619531593647248,
    0.6707909853649098,
    0.6676977461963641,
    0.6646045071331738,
    0.6690234200805887,
    0.6690234200805887,
    0.672116659038424,
    0.6699072026700716,
    0.6685815287858472,
    0.6641626157330771],
   'val_loss': [2.507451802189246,
    1.6557292326119322,
    1.3676636817441596,
    1.2499676497173267,
    1.2064521666179966,
    1.181490518801306,
    1.1689165046770278,
    1.176572942596803,
    1.1748335163413135,
    1.1766066882023494,
    1.2014401591672235,
    1.2009204000274112,
    1.215036653808614,
    1.2226964897462556,
    1.2337338884373024,
    1.2406998922800627,
    1.2640889837211915,
    1.268003778651495,
    1.2676862905471133,
    1.2827275262382565]},
  'idx': 7,
  'params': {'batch_size': 64,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'nadam'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 56, 45, 693876),
  'training_time': datetime.timedelta(0, 66, 594470),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6703490928322459,
  'history': {'acc': [0.09711634052674319,
    0.2854933154252389,
    0.41686001571156467,
    0.5019334875876147,
    0.5592752171147529,
    0.5949618831127075,
    0.6328582475164585,
    0.6630206600814456,
    0.6825765113239985,
    0.6970500487238647,
    0.7093138878759844,
    0.730085072636196,
    0.7419069709270492,
    0.7500828647400157,
    0.761573306138628,
    0.7714064737444174,
    0.7773726656415584,
    0.79206717464097,
    0.7941663898811618,
    0.8028947080963686],
   'loss': [2.9542187699006153,
    2.7214719218162524,
    2.4139150929126987,
    2.150346041753034,
    1.9413621526649019,
    1.7880945604106442,
    1.6536209094775947,
    1.535155897482418,
    1.4418807962219724,
    1.3614330868235796,
    1.3104212359425351,
    1.2284105947275554,
    1.1806721036282182,
    1.1227493515832294,
    1.0803920809780876,
    1.034080174275886,
    1.0016991264871993,
    0.9534884964599226,
    0.9302560209358691,
    0.8950433935885692],
   'val_acc': [0.29474149314481707,
    0.42421564387369093,
    0.49182501158722725,
    0.5267344213827911,
    0.5505965523523827,
    0.584180293570985,
    0.6009721620600149,
    0.6168802497242596,
    0.6261599630156924,
    0.634114009508031,
    0.6403004844474203,
    0.6429518315047223,
    0.6500220964611287,
    0.6513477678695085,
    0.6548828982274405,
    0.6623950520027437,
    0.6650463965842011,
    0.6632788353033736,
    0.6681396363585385,
    0.6703490928322459],
   'val_loss': [2.867193766837093,
    2.572989442524423,
    2.2772880972248233,
    2.053598754552419,
    1.8946073370279066,
    1.7726937255867803,
    1.6808189028825478,
    1.6055689439065572,
    1.54273054207841,
    1.49338648838645,
    1.452809148955756,
    1.4176241037227177,
    1.3862418987768916,
    1.3594239100491217,
    1.3367491180371036,
    1.3161018724943019,
    1.2996885456767062,
    1.2819541175967026,
    1.2672490116984139,
    1.2540242046684067]},
  'idx': 24,
  'params': {'batch_size': 512,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adagrad'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 12, 0, 836481),
  'training_time': datetime.timedelta(0, 49, 161198),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6690234182368745,
  'history': {'acc': [0.06872168821371691,
    0.15887747214039627,
    0.2533421722509205,
    0.34990608778108934,
    0.4307811292459024,
    0.4902220751722667,
    0.5431444041900985,
    0.5780576734223586,
    0.6062313559213115,
    0.6426914155832273,
    0.668323942283651,
    0.6978234451010192,
    0.718484145609032,
    0.7339520496487578,
    0.7506352891159142,
    0.7657717381007315,
    0.7811291571243496,
    0.7961551211194009,
    0.8057673186271207,
    0.8165948514832461],
   'loss': [2.9840331257225965,
    2.9126374165454156,
    2.792559870513496,
    2.5975920408552566,
    2.3823492816239744,
    2.1645953118135988,
    1.9650773527661471,
    1.8022420915658006,
    1.6741702629049737,
    1.5414021322442977,
    1.4334498683782773,
    1.3246529043583484,
    1.2359510803865392,
    1.1542533796250576,
    1.0765318194748539,
    1.0145073490187075,
    0.9491475274017299,
    0.888107573949175,
    0.838998587444176,
    0.8014218612028583],
   'val_acc': [0.12196199730914407,
    0.2571807330983748,
    0.38135218829693546,
    0.4564737069543568,
    0.4829871853368239,
    0.5072912044141654,
    0.5338046845744997,
    0.5612019466131701,
    0.5766681380309839,
    0.5934600089958585,
    0.5992045944052041,
    0.6199734870227522,
    0.632346444329065,
    0.6416261615186364,
    0.6451612922189723,
    0.6522315504589915,
    0.6637207226999766,
    0.6654882903547873,
    0.6672558537690554,
    0.6690234182368745],
   'val_loss': [2.9610407090113324,
    2.8849805900600702,
    2.7267688609202034,
    2.50441517147194,
    2.2753045713801434,
    2.0684799926764557,
    1.9028227433555693,
    1.7714145119823401,
    1.662169848929661,
    1.5775530953561105,
    1.5081865855986916,
    1.4443643875370709,
    1.39274922051474,
    1.3501717422159172,
    1.3129520774993864,
    1.2785573238511028,
    1.2488980753148913,
    1.2255158972223934,
    1.209703841569726,
    1.1928964925454983]},
  'idx': 15,
  'params': {'batch_size': 256,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'rmsprop'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 4, 16, 752649),
  'training_time': datetime.timedelta(0, 51, 377747),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6654882921194851,
  'history': {'acc': [0.06386034683326558,
    0.12683681336480232,
    0.2026295437511588,
    0.2810739146294666,
    0.35951828535795605,
    0.41619710558857687,
    0.47574853540823453,
    0.517732847330913,
    0.5466799237770003,
    0.5904319954239825,
    0.6130814273282844,
    0.6296541826749504,
    0.6497624565153383,
    0.6703126721718787,
    0.6785990500592379,
    0.7035686661427673,
    0.7160534744512415,
    0.734173019035602,
    0.7450005531824698,
    0.7616837917540089],
   'loss': [2.989913264159642,
    2.943510696382473,
    2.886008781354713,
    2.8096339442613782,
    2.7017967508400544,
    2.5684532598953407,
    2.4206787005219983,
    2.2722475992330673,
    2.139508215159609,
    1.9896486993126283,
    1.8658859823279823,
    1.752170054968675,
    1.6488309670332295,
    1.533771234748772,
    1.4463616138188935,
    1.363879891472381,
    1.2899023530075213,
    1.21438013166502,
    1.1483664239471016,
    1.082814660536994],
   'val_acc': [0.10517012811225966,
    0.22403888601910157,
    0.3190455140000258,
    0.3835616440726654,
    0.42996022963861,
    0.4644277520375709,
    0.4993371632422592,
    0.5192222711500529,
    0.5479452060062275,
    0.565620859560585,
    0.5828546186332095,
    0.594785682880083,
    0.6067167464158095,
    0.6102518803031372,
    0.6217410500682775,
    0.6367653533783414,
    0.649138311395801,
    0.6535572268190607,
    0.660627484690337,
    0.6654882921194851],
   'val_loss': [2.9731662876038123,
    2.9428847100531375,
    2.889603942356118,
    2.80637611154429,
    2.6890470587569846,
    2.548807592556322,
    2.4034313053564325,
    2.262306168497636,
    2.1330086867787172,
    2.011475978043034,
    1.8998697233368642,
    1.7999164502914637,
    1.709744744315592,
    1.630405936944922,
    1.5626283241382384,
    1.5017645258425192,
    1.4501793205975118,
    1.405531265226761,
    1.3666458018733187,
    1.332678266873783]},
  'idx': 22,
  'params': {'batch_size': 512,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'rmsprop'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 10, 21, 191303),
  'training_time': datetime.timedelta(0, 49, 707276),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6650463988229968,
  'history': {'acc': [0.14230471769264028,
    0.34206165067926453,
    0.4554192906334287,
    0.5271240746154402,
    0.5625897690994597,
    0.6110926969790772,
    0.6342945531012274,
    0.6557286487812628,
    0.67837807982946,
    0.6922991934527009,
    0.7127389238955711,
    0.73196331905589,
    0.7425698818402875,
    0.7598055462760301,
    0.7651088277735956,
    0.7802452767584129,
    0.7822340073776224,
    0.7896365042273319,
    0.8042205282040504,
    0.801679372418692],
   'loss': [2.8873057738942554,
    2.456345670487443,
    2.1124975249564737,
    1.8743078745131914,
    1.73336591240899,
    1.584496733128412,
    1.490181094756404,
    1.403839727420594,
    1.3116157032905817,
    1.2525188298715886,
    1.200540846243127,
    1.140063733926018,
    1.0965137242638674,
    1.0450062223750969,
    1.0117564949129527,
    0.9603803309007661,
    0.9349889981326523,
    0.8941166469858781,
    0.8603530647925627,
    0.8434841672355543],
   'val_acc': [0.3495360139956582,
    0.4551480338207874,
    0.5006628370738061,
    0.542642510074248,
    0.574900574590377,
    0.5925762264853919,
    0.6080424218013442,
    0.617764030285657,
    0.6283694213594528,
    0.6363234646647997,
    0.6411842689069561,
    0.6456031818543712,
    0.650905877391269,
    0.6495802035070446,
    0.652673442570235,
    0.655324790338684,
    0.6588599206966159,
    0.6597437032860989,
    0.6619531597598064,
    0.6650463988229968],
   'val_loss': [2.673231249934427,
    2.2164748148005775,
    1.953907507160202,
    1.7876988204769786,
    1.6756007606036043,
    1.587836887159023,
    1.5209013518642336,
    1.4668877766925679,
    1.4218356464481228,
    1.386853240466571,
    1.3555491600108263,
    1.3298485757922156,
    1.309063326611858,
    1.2898734237470302,
    1.2719822808593129,
    1.2569828592356043,
    1.244744826827595,
    1.2319648459691996,
    1.2216040777833495,
    1.2120133621525142]},
  'idx': 3,
  'params': {'batch_size': 64,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adagrad'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 52, 31, 664706),
  'training_time': datetime.timedelta(0, 61, 304579),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 18,
  'best_score': 0.6610693788296661,
  'history': {'acc': [0.11799801129416831,
    0.3253784113010113,
    0.45099988981131467,
    0.5210473980874386,
    0.5726439069613688,
    0.6100983318999622,
    0.6400397748015175,
    0.6626892057970313,
    0.684123301450725,
    0.7065517623679254,
    0.7166059000454428,
    0.730416528731601,
    0.7443376423745982,
    0.7579273010087848,
    0.7706330793165473,
    0.7778146062723353,
    0.7828969175928061,
    0.787426803826153,
    0.7950502707936885,
    0.803336647982993],
   'loss': [2.927821076913323,
    2.6003702721033632,
    2.236818864534947,
    1.9650000498421534,
    1.7832249840403536,
    1.632806515432882,
    1.5258075372380666,
    1.4282534877156279,
    1.3438402744646614,
    1.277675401517771,
    1.2089730878256129,
    1.1592621592550223,
    1.1076056342298268,
    1.0510709060909813,
    1.0255488325661553,
    0.985136722743926,
    0.9432522665333477,
    0.9214770124045779,
    0.886488253227875,
    0.8643278820569206],
   'val_acc': [0.34555899131577256,
    0.4485196636621793,
    0.5046398580680103,
    0.5422006159875965,
    0.574900572852018,
    0.5899248796914776,
    0.6040653972250668,
    0.6120194405304137,
    0.6212991609069767,
    0.6270437477386162,
    0.6385329199796012,
    0.6447193998706799,
    0.6442775061000937,
    0.6460450712790596,
    0.6469288545796895,
    0.6504639842264746,
    0.6557666797633726,
    0.6553247902333289,
    0.6610693788296661,
    0.6579761404776225],
   'val_loss': [2.7851352394549025,
    2.408243022342198,
    2.0798385003157756,
    1.86787582423589,
    1.729888098565446,
    1.6318487181792047,
    1.5583173427442718,
    1.499398066572947,
    1.4511128730917209,
    1.41204749900474,
    1.3804201689432323,
    1.3531030508674355,
    1.3288597238311397,
    1.3077068899882034,
    1.2898158560903317,
    1.2739482719814994,
    1.2606659822636772,
    1.2466850195424093,
    1.2351279302977167,
    1.2258727064819606]},
  'idx': 17,
  'params': {'batch_size': 256,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adagrad'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 5, 58, 874615),
  'training_time': datetime.timedelta(0, 51, 91336),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6385329210331521,
  'history': {'acc': [0.06330792177824562,
    0.15954038226173775,
    0.2890288365992271,
    0.3716716384798134,
    0.42857142860435565,
    0.4644790631165861,
    0.505137553825232,
    0.5332007513113699,
    0.5617058888191334,
    0.5874489006607878,
    0.6119765771474512,
    0.6370566788792886,
    0.6561705888786207,
    0.6778256546567256,
    0.701469450975015,
    0.7037896364515703,
    0.7238979118724591,
    0.7465473427823625,
    0.7546127499657933,
    0.7641144624837471],
   'loss': [2.9839893711323158,
    2.893517601728413,
    2.6799801573702253,
    2.4307141169673527,
    2.221161346226921,
    2.0691651896526375,
    1.9246955385736018,
    1.8111439225766135,
    1.7053613427883616,
    1.619435191562776,
    1.5411658130578738,
    1.4553800383637248,
    1.3778169203721256,
    1.3003689184753608,
    1.2389790292287213,
    1.1892571337064213,
    1.1349288240847253,
    1.078627946152633,
    1.019268643690096,
    0.9833012873817446],
   'val_acc': [0.19929297396133708,
    0.3247901017008447,
    0.3893062306145783,
    0.4259832080781223,
    0.4582412725942513,
    0.4882898805313178,
    0.5156871406999353,
    0.5346884663738195,
    0.5461776400370983,
    0.5669465308899485,
    0.5740167916058124,
    0.5802032697321934,
    0.5978789220222898,
    0.6009721610854802,
    0.6102518782750516,
    0.6173221389909155,
    0.6235086171172964,
    0.6319045517173848,
    0.6283694213594528,
    0.6385329210331521],
   'val_loss': [2.9562270721396775,
    2.818269401222428,
    2.524661263374433,
    2.271147070445163,
    2.0934192524301865,
    1.962428108180354,
    1.8536454820021875,
    1.765617143196497,
    1.688332780135505,
    1.622891705224749,
    1.5689026974915299,
    1.5176467140111785,
    1.4751137386525146,
    1.4366754590543138,
    1.4009305121252824,
    1.3723432041088985,
    1.3478609831839918,
    1.3244996395882283,
    1.3038228897726014,
    1.285219462793378]},
  'idx': 6,
  'params': {'batch_size': 64,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adamax'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 55, 42, 319410),
  'training_time': datetime.timedelta(0, 63, 373935),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6319045521124664,
  'history': {'acc': [0.06629101759593087,
    0.13468125069300096,
    0.22815158556882958,
    0.3132250581198852,
    0.38415644688377615,
    0.4348690755300635,
    0.484034913292301,
    0.5236990390765892,
    0.5480057454717685,
    0.5686664459863666,
    0.5952933380029805,
    0.6158435534619584,
    0.6387139544501751,
    0.6517511879313836,
    0.6686553974044137,
    0.685559606864273,
    0.7015799361096602,
    0.7179317204428831,
    0.7293116783597906,
    0.741796486734119],
   'loss': [2.9879882237144946,
    2.9287971159777264,
    2.822616959339926,
    2.639898519465415,
    2.4440801447892686,
    2.237822736880882,
    2.067229178041111,
    1.9207944748559955,
    1.8107517349853026,
    1.7261567439259566,
    1.6320747321461488,
    1.553981964330807,
    1.4726040297062433,
    1.4193702241673996,
    1.363524614717536,
    1.2894179313805285,
    1.2343252462229668,
    1.1840585668157337,
    1.1393741004415117,
    1.0900827748592934],
   'val_acc': [0.11268228014261913,
    0.2240388866841556,
    0.3066725587876455,
    0.3826778607588661,
    0.43084401245197257,
    0.46531153315208257,
    0.4953601411945041,
    0.5240830752078379,
    0.539549270378927,
    0.5594343796431676,
    0.5678303129526561,
    0.5797613789115498,
    0.5899248786906042,
    0.594343792349166,
    0.6027397269492544,
    0.6106937702546013,
    0.6195315961494311,
    0.622624834922895,
    0.6266018552849686,
    0.6319045521124664],
   'val_loss': [2.9716857583766165,
    2.910665620284643,
    2.7492828299606265,
    2.536985815493238,
    2.3220396338971483,
    2.1224979593645297,
    1.9647538666170992,
    1.848934389140508,
    1.7641389113424943,
    1.6923045574749565,
    1.6293623552246532,
    1.5751787974041118,
    1.5287463426379255,
    1.489206822372568,
    1.4527920588317615,
    1.420891170316284,
    1.392306529732863,
    1.3668972782155238,
    1.3447776337508548,
    1.3244110192022016]},
  'idx': 13,
  'params': {'batch_size': 128,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adamax'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 2, 24, 818928),
  'training_time': datetime.timedelta(0, 55, 435886),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6199734856004584,
  'history': {'acc': [0.06253452660046076,
    0.11170036460306615,
    0.18561484923897204,
    0.25499944768351307,
    0.32327919572496294,
    0.37664346485990885,
    0.42956579384481347,
    0.47232350022828906,
    0.5062424044374102,
    0.5296652306510445,
    0.5647994699550326,
    0.5856811404491628,
    0.6021434097721516,
    0.6187161642100294,
    0.6359518287511389,
    0.66291017588193,
    0.6692078225573919,
    0.6821345709961509,
    0.7064412773518177,
    0.7031267264800471],
   'loss': [2.991856617896468,
    2.9509190832129,
    2.902338223620128,
    2.8219374713153815,
    2.7052554630361305,
    2.540180950556759,
    2.37624096786129,
    2.225788321472402,
    2.1001569267768594,
    1.989565868059478,
    1.8773916289366723,
    1.795156397762621,
    1.7157592706793041,
    1.6303422834949326,
    1.5605729600761808,
    1.4830269385221821,
    1.4283488733098595,
    1.3698734074267955,
    1.3138110708919049,
    1.2779893769887043],
   'val_acc': [0.09323906316082403,
    0.18559434443672726,
    0.26690234193167645,
    0.34114007942190855,
    0.37604949133774374,
    0.41758727338584817,
    0.44984533754640377,
    0.4688466625091411,
    0.49580203252875377,
    0.510826336905538,
    0.5267344206716442,
    0.539549273170837,
    0.5510384429359774,
    0.5665046375407827,
    0.5771100286145785,
    0.5881573127478137,
    0.6014140526436095,
    0.6093680959489564,
    0.6129032238310436,
    0.6199734856004584],
   'val_loss': [2.9782139796679705,
    2.9519603712812654,
    2.9009298543934268,
    2.80442016913835,
    2.654550913314718,
    2.477880213885406,
    2.308781648177359,
    2.1667501294291553,
    2.0543352115275426,
    1.9603039175337835,
    1.8776473996916336,
    1.80540941533819,
    1.7399048351683135,
    1.681649377216452,
    1.6311226397344512,
    1.5850672033894024,
    1.546185193417929,
    1.5093788988970573,
    1.4778596839175728,
    1.4492006780192013]},
  'idx': 20,
  'params': {'batch_size': 256,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adamax'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 8, 35, 504133),
  'training_time': datetime.timedelta(0, 52, 580008),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6151126821221264,
  'history': {'acc': [0.06120870622277668,
    0.13081427466660597,
    0.23378632193691176,
    0.31333554301745536,
    0.37145066846406155,
    0.41862777590014083,
    0.4544249255016303,
    0.49508341620698343,
    0.518616727397213,
    0.5442492542589795,
    0.5723124516726779,
    0.5988288586172079,
    0.614849187967962,
    0.6342945530814712,
    0.6687658821702754,
    0.6783780797899474,
    0.6958347143698577,
    0.7155010495814365,
    0.7266600375122266,
    0.7402496962254382],
   'loss': [2.9903435154180342,
    2.9107607918514784,
    2.6956412916011723,
    2.4598864632645387,
    2.2415082631170677,
    2.068359159551369,
    1.9288687310341563,
    1.8132485347668146,
    1.7140291387264852,
    1.6283589961080707,
    1.5309557824118494,
    1.449565114441293,
    1.3838743108923877,
    1.3351854061819601,
    1.2541022564795594,
    1.1973194210730314,
    1.1379922996725302,
    1.0757308801871324,
    1.0448207792926527,
    0.9889349754891097],
   'val_acc': [0.10517012807604385,
    0.23862129913406777,
    0.325231992784876,
    0.37251436141440153,
    0.4189129471515482,
    0.44631020763623097,
    0.4710561200363995,
    0.48917366312080074,
    0.5183384884683843,
    0.5298276622370182,
    0.550154661900482,
    0.5634114006373717,
    0.5771100306690029,
    0.5775519224641811,
    0.5921343351906504,
    0.6027397262644463,
    0.6062748562272966,
    0.6067167474166829,
    0.6098099864798734,
    0.6151126821221264],
   'val_loss': [2.9684582860713187,
    2.8047796736973027,
    2.5129065317650365,
    2.2601592627132145,
    2.056282380698689,
    1.9198561169328914,
    1.8195604111207464,
    1.7355626367964263,
    1.6652642003846179,
    1.602500182519273,
    1.5458586169089883,
    1.4947383259441596,
    1.459542709749671,
    1.4282839171822177,
    1.3999188784946133,
    1.371693556518403,
    1.346172482736649,
    1.3299398575532873,
    1.3145806503359112,
    1.2957249298276914]},
  'idx': 4,
  'params': {'batch_size': 64,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adadelta'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 53, 32, 971302),
  'training_time': datetime.timedelta(0, 65, 77108),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.6102518771161456,
  'history': {'acc': [0.05988288588460513,
    0.09004529898301131,
    0.1415313223790311,
    0.19058667522888134,
    0.23632747773379467,
    0.29400066236358574,
    0.3439398963350497,
    0.38039995623733336,
    0.4298972491894805,
    0.4618274221471918,
    0.4825986077042393,
    0.5263506798780553,
    0.5493315652238375,
    0.5829190140718281,
    0.6004861340958985,
    0.621478289711503,
    0.6341840673870651,
    0.6556181648188224,
    0.6633521160188004,
    0.6726328583201471],
   'loss': [2.995063823911317,
    2.963822157287345,
    2.9333515795485483,
    2.8989105774096897,
    2.853379928029159,
    2.786528409534944,
    2.7037741022865465,
    2.600265018637833,
    2.4712951233895124,
    2.344098453929498,
    2.2282679664694847,
    2.1042160471293587,
    2.0057828908545603,
    1.8890028158849992,
    1.807820562262151,
    1.7356526193770503,
    1.6538306400797247,
    1.5835247967417314,
    1.523643669430399,
    1.4658272700894013],
   'val_acc': [0.10163499811648585,
    0.13477684521222083,
    0.17808219137257092,
    0.2319929303779994,
    0.3053468849165904,
    0.3756075997005982,
    0.41758727303027476,
    0.44410075141274186,
    0.4622182948395472,
    0.4816615103858791,
    0.502430400172009,
    0.5236411833731516,
    0.5386654898702071,
    0.5461776418808125,
    0.5603181601255487,
    0.5802032701536137,
    0.5846221848657265,
    0.5978789240503753,
    0.6049491847662393,
    0.6102518771161456],
   'val_loss': [2.9819366945611447,
    2.9692081638105243,
    2.9504676606873206,
    2.9205854691291773,
    2.874020582693001,
    2.8054106109078036,
    2.711177599624991,
    2.591601722808312,
    2.4555609645540084,
    2.3217548454856116,
    2.202346697666979,
    2.0937325226433554,
    1.999331865909086,
    1.9146096502001497,
    1.84125381968899,
    1.7762893537899798,
    1.71898501208015,
    1.6669416027262944,
    1.6203888889123947,
    1.5797848612451997]},
  'idx': 27,
  'params': {'batch_size': 512,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adamax'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 14, 30, 231284),
  'training_time': datetime.timedelta(0, 50, 139738),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.5894829863949892,
  'history': {'acc': [0.05734173022436972,
    0.08982432882567293,
    0.15490001107319826,
    0.20682797494441446,
    0.2582035135391759,
    0.302066070156168,
    0.345818141750467,
    0.3731079439229958,
    0.4111147939750548,
    0.42978676393629767,
    0.4614959673392366,
    0.4850292785623932,
    0.5106618053484274,
    0.5319854162461375,
    0.5526461167014669,
    0.5819246494471072,
    0.5979449786529819,
    0.6091039666627971,
    0.6295436970595694,
    0.6430228705986233],
   'loss': [2.992856185826896,
    2.963213896016471,
    2.877408374305483,
    2.751642284597722,
    2.6048282754051133,
    2.4644541544698773,
    2.3227608359040457,
    2.2018102510319717,
    2.1023317423726615,
    2.0112725309397868,
    1.905581975323023,
    1.8268555026464786,
    1.7505620896598935,
    1.6713204208941317,
    1.5961521528739453,
    1.5279281056660892,
    1.4688887086023665,
    1.4129490321439322,
    1.3647396315291316,
    1.3234641890066945],
   'val_acc': [0.09589041094573472,
    0.16394167018447678,
    0.21078214742049042,
    0.26115775515271455,
    0.3296509064039297,
    0.35749005782778054,
    0.39372514385171975,
    0.4171453821832924,
    0.4414494032492113,
    0.4626601856733601,
    0.47989394616827835,
    0.49889527169729936,
    0.5165709233420959,
    0.5240830750629747,
    0.5413168358476195,
    0.5567830308738452,
    0.5576668134633282,
    0.5753424652529879,
    0.5824127269697252,
    0.5894829863949892],
   'val_loss': [2.984579733605412,
    2.9462490500046834,
    2.8000959217469354,
    2.6462234153823414,
    2.474584402260504,
    2.3184155688789425,
    2.1684941926257464,
    2.063232479820302,
    1.9819694211195382,
    1.9034225204272235,
    1.8245592449178119,
    1.7576885345600581,
    1.6958242061124458,
    1.6411407392635,
    1.5911670227362633,
    1.547624222535872,
    1.5128656682112656,
    1.476492849799631,
    1.445417882170869,
    1.4232908715413726]},
  'idx': 11,
  'params': {'batch_size': 128,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adadelta'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 0, 33, 268617),
  'training_time': datetime.timedelta(0, 55, 544052),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.47989394594439877,
  'history': {'acc': [0.05402717934601373,
    0.06209258645906289,
    0.09137111922240154,
    0.11523588560171752,
    0.14363053808020249,
    0.17810186733199593,
    0.20318196890084417,
    0.2284830406895923,
    0.25091150163313447,
    0.28726107623284464,
    0.29897248935612536,
    0.3332228483155793,
    0.3560932494009309,
    0.37509667449610734,
    0.3980775605399442,
    0.4090155784582752,
    0.43332228487321073,
    0.4549773506118031,
    0.4611645122448155,
    0.4792840570333241],
   'loss': [2.9975263426520127,
    2.9824727634450583,
    2.9664487963604698,
    2.937306820865547,
    2.898379574137068,
    2.8469189887625803,
    2.783120759744405,
    2.70703713149211,
    2.618103554414499,
    2.512707232190896,
    2.4347857083685294,
    2.346080004214029,
    2.2776718878269246,
    2.2162288582373715,
    2.1406484378291655,
    2.0832324215567923,
    2.0241939590828366,
    1.942163857277316,
    1.9001018057974803,
    1.8582598001914004],
   'val_acc': [0.05612019475811201,
    0.06584180288355908,
    0.11621741082978575,
    0.15422006217426185,
    0.19089703962463644,
    0.20724701787906044,
    0.24171453778900723,
    0.27264692911888927,
    0.29960229844052444,
    0.3199292976430597,
    0.3455589930804704,
    0.3663278828666003,
    0.3835616430059451,
    0.3985859491474271,
    0.4091913387989292,
    0.4237737508142516,
    0.4387980551910358,
    0.44896155461451676,
    0.4723817946449403,
    0.47989394594439877],
   'val_loss': [2.992134792321138,
    2.984430016008985,
    2.9682552416241386,
    2.9245662700798043,
    2.882604991973526,
    2.8181235882965168,
    2.7349005387095926,
    2.629595819533108,
    2.5137422051094718,
    2.4089008079289442,
    2.329131481029057,
    2.2502932846677868,
    2.1858626275899082,
    2.1213934644892105,
    2.058425847271402,
    1.9975935293940175,
    1.9466251079034236,
    1.8991322666472479,
    1.855170163231722,
    1.8166565697539765]},
  'idx': 18,
  'params': {'batch_size': 256,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adadelta'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 6, 49, 972894),
  'training_time': datetime.timedelta(0, 52, 560241),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.34997790568548126,
  'history': {'acc': [0.05236990388584471,
    0.05712076003986648,
    0.06363937676729992,
    0.07943873617968077,
    0.08043310127032026,
    0.09391227478220929,
    0.10926969386674254,
    0.11799801132297953,
    0.1347917357510907,
    0.14738702881060942,
    0.1653960888612443,
    0.19069716065657782,
    0.21069495127209245,
    0.23058225624917714,
    0.2512429563258448,
    0.2640592196694712,
    0.28880786657359714,
    0.3000773398761077,
    0.3118992379792763,
    0.3226162851278248],
   'loss': [3.0002462842058524,
    2.9897431096587046,
    2.983393532467763,
    2.9753762549846137,
    2.968261918683379,
    2.958873915048004,
    2.948315138843702,
    2.9330797467544163,
    2.914220487040165,
    2.8900539082409256,
    2.8596241061377428,
    2.812909311042308,
    2.761358685314873,
    2.7130937961936197,
    2.651982491404401,
    2.579587602494185,
    2.509078378822774,
    2.450204225998928,
    2.396646423532665,
    2.3477812179767312],
   'val_acc': [0.058771542184156915,
    0.0671674767710759,
    0.09544851998352025,
    0.11312417141760654,
    0.1162174104742123,
    0.12903225872298546,
    0.14714980110282452,
    0.1559876269910697,
    0.17498895265836922,
    0.19487406090856704,
    0.21475916916534957,
    0.22845779894676244,
    0.25408749474633124,
    0.2748563848946193,
    0.29474149315140175,
    0.3133009282351067,
    0.31683605859303865,
    0.3274414489556876,
    0.33318603755202486,
    0.34997790568548126],
   'val_loss': [2.9930523123511477,
    2.9885854540912122,
    2.984283198627602,
    2.979957563604236,
    2.973377134227879,
    2.9648550514409986,
    2.951538743985421,
    2.931944290410614,
    2.9074630811053255,
    2.8766478739952124,
    2.835302045268397,
    2.785062885157998,
    2.7242630343494088,
    2.6583650472466522,
    2.5862641651652525,
    2.506665753938959,
    2.4324599988577487,
    2.365566442774504,
    2.3057783051396807,
    2.251990810728052]},
  'idx': 25,
  'params': {'batch_size': 512,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'adadelta'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 12, 49, 998097),
  'training_time': datetime.timedelta(0, 49, 839988),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 19,
  'best_score': 0.0737958462300603,
  'history': {'acc': [0.049828748205441455,
    0.05148602364544262,
    0.050823113468948244,
    0.05336426914153132,
    0.05667882002071048,
    0.06087725113247155,
    0.060766766103193016,
    0.059219975693293556,
    0.06176113136752299,
    0.06010385592999135,
    0.05767318528504046,
    0.06673295768588014,
    0.06463374212876484,
    0.06264501160175125,
    0.06507568224587897,
    0.06629101756794284,
    0.06507568224587897,
    0.07037896365124854,
    0.06761683791928523,
    0.06839023312423495],
   'loss': [3.0046592012135025,
    2.999359990889006,
    2.9966819000802585,
    2.992653450219358,
    2.9929281094761504,
    2.990156297511968,
    2.9906928037730465,
    2.9892383949912826,
    2.9881282609782156,
    2.9879318251160414,
    2.988234050839451,
    2.985489077863608,
    2.985132321219802,
    2.983988742066568,
    2.9851007004256225,
    2.9838929989498344,
    2.9842720913394434,
    2.9822988282803107,
    2.982123443875151,
    2.981438173956403],
   'val_acc': [0.049491824997877895,
    0.05081749888210237,
    0.048166151143284534,
    0.04860804243802603,
    0.04772425984854304,
    0.04949182501927815,
    0.05258506409069947,
    0.05832965092233889,
    0.055236411850917565,
    0.06274856386152296,
    0.0618647812638091,
    0.06363234644277507,
    0.06319045514803358,
    0.06539991162174105,
    0.06805125939019001,
    0.06981882458561772,
    0.06716747680893789,
    0.07202828104286346,
    0.07291206364057731,
    0.0737958462300603],
   'val_loss': [2.9987277443093956,
    2.9961967352371115,
    2.9946737014451914,
    2.9939548228622486,
    2.9933472485653976,
    2.993024117326083,
    2.992679408099132,
    2.992452668553688,
    2.992242542031693,
    2.992093567610103,
    2.991852316664638,
    2.991715179836123,
    2.9915092886942865,
    2.991317711517024,
    2.9911061585924026,
    2.990788529733615,
    2.9906189134044032,
    2.9902667537812264,
    2.9900382569128046,
    2.989879977634614]},
  'idx': 2,
  'params': {'batch_size': 64,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'sgd'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 51, 31, 755890),
  'training_time': datetime.timedelta(0, 59, 906892),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 18,
  'best_score': 0.06495802031712251,
  'history': {'acc': [0.05192796376646685,
    0.05424814941115622,
    0.05413766435100853,
    0.05214893383058036,
    0.05148602366149459,
    0.05402717932173,
    0.054469119438844124,
    0.05844658049739878,
    0.05756270027634136,
    0.056126394884607536,
    0.054800574526679725,
    0.056568334993078305,
    0.056899790100670165,
    0.058557065553019,
    0.05623687990730065,
    0.05623687993816981,
    0.06297646671098947,
    0.059330460744797885,
    0.06043531104416863,
    0.061871616405033296],
   'loss': [3.0026742133655913,
    3.0016661673906087,
    2.9982508681331757,
    2.9970321150303363,
    2.9967354394469967,
    2.99560186112471,
    2.994600353208303,
    2.9931122758035276,
    2.992026807168428,
    2.9917471502883806,
    2.9925174385180804,
    2.990008775864684,
    2.989857562588481,
    2.9891353033350496,
    2.988328342871855,
    2.9882063508731807,
    2.987560966188454,
    2.9880837938453912,
    2.9863237983521116,
    2.9858382723105175],
   'val_acc': [0.05523641183774818,
    0.05435262924991136,
    0.05435262924991136,
    0.05479452054300669,
    0.05832965090093863,
    0.05877154219568013,
    0.05832965090093863,
    0.05965532478516312,
    0.05965532478516312,
    0.060980998669387596,
    0.05744586831145565,
    0.05788775960455097,
    0.05921343348877545,
    0.05877154219403395,
    0.060097216078258436,
    0.06098099866609525,
    0.06274856384506122,
    0.06230667255031973,
    0.06495802031712251,
    0.06451612902238102],
   'val_loss': [2.9982407914817832,
    2.9970385635315293,
    2.9961691068117458,
    2.995471167553857,
    2.99487203936423,
    2.994430930436421,
    2.9941157942791716,
    2.9937832106861246,
    2.9935938124629287,
    2.99338971773081,
    2.9932142951769793,
    2.993033580004616,
    2.992869617451201,
    2.9927397432761333,
    2.9925360044335245,
    2.992407335520739,
    2.99228479890659,
    2.9921193185023274,
    2.9920482769299017,
    2.9919132744902486]},
  'idx': 9,
  'params': {'batch_size': 128,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'sgd'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 14, 58, 46, 459709),
  'training_time': datetime.timedelta(0, 53, 135129),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 18,
  'best_score': 0.05965532476870138,
  'history': {'acc': [0.03900121536413328,
    0.03977461054479927,
    0.04264722131921196,
    0.0451883769860328,
    0.04739807757160345,
    0.04518837697944737,
    0.0499392332561226,
    0.053474754181922754,
    0.053585239217786705,
    0.053253784123365684,
    0.05281184400625156,
    0.05457960446812266,
    0.05579493979018652,
    0.05181747874274476,
    0.056015909879612746,
    0.05977240084246445,
    0.05579493980335736,
    0.056126394878022116,
    0.057231245170807445,
    0.05767318531220532],
   'loss': [3.009559025409906,
    3.007526388748906,
    3.006664895284322,
    3.004162299410439,
    3.002507479832773,
    3.001970425243207,
    2.9992433243358403,
    2.9980750078655003,
    2.997107824084904,
    2.9963849118738173,
    2.9957314764329865,
    2.9949535307548625,
    2.993905385941145,
    2.9947857623626586,
    2.991965706632119,
    2.9924341863906787,
    2.9928199226007237,
    2.9919606538493784,
    2.990026339366678,
    2.990191551087746],
   'val_acc': [0.039328325215531194,
    0.039328325215531194,
    0.03888643391914353,
    0.04374723816294613,
    0.046840477221198065,
    0.04904993369655171,
    0.056562085705510926,
    0.05744586829828626,
    0.056120194414061776,
    0.054352629233449634,
    0.05479452052819112,
    0.05523641182293262,
    0.05435262923180346,
    0.057003976998606244,
    0.0583296508844769,
    0.05921343347395989,
    0.05877154217921839,
    0.05832965088283073,
    0.05965532476870138,
    0.05921343347395989],
   'val_loss': [3.0015260673865143,
    3.0002089521287556,
    2.9990850051208984,
    2.998081360567053,
    2.9972402826116196,
    2.996508318820571,
    2.9959048937234107,
    2.99534900073609,
    2.994941991986398,
    2.994560383296466,
    2.994182169569102,
    2.9938850763146454,
    2.993613011848595,
    2.9934034363337445,
    2.9931715622024098,
    2.9929953350517216,
    2.9928221182121515,
    2.992640429874779,
    2.992480922156185,
    2.9923752988275045]},
  'idx': 16,
  'params': {'batch_size': 256,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'sgd'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 5, 8, 130921),
  'training_time': datetime.timedelta(0, 50, 743258),
  'weights_file': '/tmp/weights.hdf5'},
 {'__type__': 'keras',
  'best_epoch': 1,
  'best_score': 0.051259390183428566,
  'history': {'acc': [0.048944867978210196,
    0.05281184401818763,
    0.04728759250157763,
    0.05259087394440178,
    0.05314329910602323,
    0.052369904013848834,
    0.05469008953073989,
    0.05325378405257241,
    0.054027179453026826,
    0.05181747867936008,
    0.05270135900372629,
    0.05203844876796313,
    0.05546348469823503,
    0.05071262844666673,
    0.054027179325022705,
    0.055021544521028934,
    0.05424814940621716,
    0.05038117338146851,
    0.0557949398461626,
    0.0562368798430928],
   'loss': [3.0044656791366697,
    3.0031691846920237,
    3.0038135018273358,
    3.0024152392906953,
    3.001433442250337,
    3.0018797444449037,
    3.001477841395276,
    3.0008231749812246,
    2.9997022802850024,
    3.0000754213506986,
    2.997207492402266,
    2.997306445121449,
    2.997153685351338,
    2.997826067100848,
    2.996878111353238,
    2.9961015073097474,
    2.996708482447915,
    2.9975471870897357,
    2.9968815324530658,
    2.9945925865736],
   'val_acc': [0.04993371629755791,
    0.051259390183428566,
    0.051259390183428566,
    0.05081749888868707,
    0.0503756075922994,
    0.05037560759065323,
    0.049933716295911736,
    0.04816615111694576,
    0.04860804241004108,
    0.049049933704782576,
    0.048608042408394905,
    0.048608042408394905,
    0.048608042406748736,
    0.050375607587360885,
    0.0490499337031364,
    0.048608042408394905,
    0.04816615111365342,
    0.04816615111365342,
    0.04772425981891192,
    0.04772425981891192],
   'val_loss': [3.001092539179183,
    3.0006294365325332,
    3.0001965154866124,
    2.9998018207457338,
    2.9994405253608827,
    2.9991244666090697,
    2.9988083973217474,
    2.9985158342310037,
    2.9982591565604553,
    2.998006345527234,
    2.997778894202771,
    2.9975633127100427,
    2.9973501138227787,
    2.997145486520657,
    2.9969745717952594,
    2.9967931248264086,
    2.996649048363167,
    2.996509368894799,
    2.996379214475179,
    2.9962564294337173]},
  'idx': 23,
  'params': {'batch_size': 512,
   'cnn_filter_size': 5,
   'cnn_num_filters': 50,
   'embedding_preload': False,
   'embedding_train': True,
   'epochs': 20,
   'optimizer': 'sgd'},
  'session': '20181117145020',
  'starting_time': datetime.datetime(2018, 11, 17, 15, 11, 10, 899030),
  'training_time': datetime.timedelta(0, 49, 937017),
  'weights_file': '/tmp/weights.hdf5'}]

from datetime import datetime

#%%############################################################################
   
# See how optimizers are ordered
   
[ (r['params']['optimizer'], r['best_score'], r['best_epoch'], r['training_time'].seconds ) for r in best_reports0_topN ]



#%%############################################################################

# Best params0 for finding best optimization combination

best_params0 = best_report0['params']

# From training in Collab 2018-11-17

best_params0 = {'batch_size': 512,
  'cnn_filter_size': 5,
  'cnn_num_filters': 50,
  'embedding_preload': False,
  'embedding_train': True,
  'epochs': 20,
  'optimizer': 'nadam'}

#%%############################################################################

#
# OPTIMIZATION HYPERPARAMETERS

EPOCHS = 20
OPTIMIZER = 'nadam'
BATCH_SIZE = 512

#%%############################################################################

#
# This grid is to explore embedding boolean params
#

# Grid of 4
grid1  = {'embedding_preload': [True, False],
         'embedding_train': [True, False],
         'cnn_filter_size': [5],
         'cnn_num_filters': [50],
         'batch_size': [BATCH_SIZE],
         'epochs': [EPOCHS],
         'optimizer': [OPTIMIZER]
         }

#%%############################################################################

reports1, models1 = run_keras_training_session_from_grid(grid1)







#%%############################################################################
#%%############################################################################
#%%############################################################################
#%%############################################################################

reports, models = run_keras_training_session_from_grid(grid)





#%%############################################################################

best_report = sort_models_by_score(reports)[0]

plot_epochs(best_report, renderer='pyplot')

#%%############################################################################

###############################################################################
#
# Train best (hype)parmas combination
#
###############################################################################

# TODO: change this to single fit, not a session
best_model_params = best_report['params'].copy()
best_model_params['epochs'] = 20

rr, mm = run_keras_training_session([best_model_params], session_id='BEST')
# TODO: change this to modelA
best_report = rr[0]
best_model = mm[0]

#%%############################################################################

# Save best model
best_model_definition_filepath = save_model(best_model)
best_model_weights_filepath = best_report['weights_file']

#%%############################################################################

#
# Load best weights
#

best_model.load_weights(best_model_weights_filepath)

#%%############################################################################

#
# Evaluate model with test partition
#

best_model_loss, best_model_acc = best_model.evaluate(X_seqs_test, Y_1hot_test)

#%%############################################################################

###############################################################################
#
# Train size=1 cnn filter
#
###############################################################################

# Build a new model with embedding of best model

paramsB = {'embedding_preload': True,
           'embedding_train': False,
           'embedding_matrix': best_model.layers[0].get_weights()[0],
           'cnn_filter_size': 1,
           'cnn_num_filters': best_model.layers[1].filters,
           'batch_size': 128,
           'epochs': 20,
           'optimizer': 'rmsprop'
           }

modelB = create_keras_model(**paramsB)

reportB = fit_keras_model(modelB, paramsB, X_seqs_train, Y_1hot_train, session='modelB',
                save_weights=True,
                weights_filepath='/tmp/modelB.weights.hdf5')

#%%############################################################################

#
# Evaluate model with test partition
#

modelB_loss, modelB_acc = modelB.evaluate(X_seqs_test, Y_1hot_test)


###############################################################################
#
# (C) Text sequences -> Embeddings -> Average -> SVM
#
###############################################################################





###############################################################################
###############################################################################
###############################################################################
#%%

#
# Wrap Grid Search
#
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# Tried but did not work at 1st try, I went for my own grid search


#cnn_filter_size = [2,3,4]
#batch_size = [128]
#epochs = [20]
#param_grid = dict(cnn_filter_size=cnn_filter_size, batch_size=batch_size, epochs=epochs)

#model = KerasClassifier(build_fn=create_model, verbose=1)

#grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
#grid_result = grid.fit(X, Y)


#history = model.fit(X_seqs_train, Y_1hot_train,
#                    epochs= 20,
#                    batch_size = 128,
#                    validation_split = 0.2
#                    )

#history = grid.fit(X_seqs_train, Y_1hot_train)


#%%############################################################################

###############################################################################
#
# (D) Text sequences -> TFIDF -> SVM
#
###############################################################################


#%%############################################################################

parameters = {'kernel':('linear', 'rbf'),
              'C':[0.1, 1, 10],
              'gamma': ['auto', 'scale']}

clf_svm = svm.SVC(gamma='scale', decision_function_shape='ovo')
#clf_svm = svm.LinearSVC()

clf_D = GridSearchCV(clf_svm, parameters, cv=5)

#%%############################################################################

clf_D.fit(X_tfidf_train, Y_train)

#%%############################################################################

svm_score = clf_D.score(X_tfidf_test, Y_test)
svm_score
