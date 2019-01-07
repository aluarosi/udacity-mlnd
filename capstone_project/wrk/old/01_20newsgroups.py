#%%

from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#%%
from os import path

# Project's directory

DIR='/Volumes/PENDRIVENEW/live!/PROJECTS/UDACITY_ML_NANODEGREE/my_MLND_projects/nlp-cnn/images'

#%%

# Import raw data
from sklearn.datasets import fetch_20newsgroups
raw_data_train = fetch_20newsgroups(subset='train', random_state = 42)
raw_data_test = fetch_20newsgroups(subset='test', random_state = 42)

tr = raw_data_train
te = raw_data_test

#%%

print(f"Keys: {list(tr.keys())}")
print()

target_names = tr.get('target_names')
print("\n".join(target_names))
print()

# Number of docs

tr_N = len(tr.data)
te_N = len(te.data)
N = tr_N + te_N

print(f"Training dataset: {tr_N} documents.")
print(f"Test dataset: {te_N} documents.")
print(f"Total in 20 Newsgroup dataset: {N} documents.")

#%%

# Split per class

named_target = map(lambda x: tr.target_names[x]  , tr.target)

mydf = pd.DataFrame({'count': tr.target, 'class name': list(named_target) }  )

classes_summary = mydf.groupby('class name', as_index=False).count()

import seaborn as sns

sns.set(style="whitegrid", rc={'figure.figsize':(11.7,8.27)})
ax = sns.barplot(y="class name", x="count", color="grey", data=classes_summary)
ax.set_title("20 Newsgrous - Number of documents by class") 

#%%
from math import log
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


L = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

H(L)

#%%

# Entropy of classes

entropy = H(classes_summary['count'].values)
 
print(f'Entropy of training set: {entropy:.3f} bits.')
    
#%%

# Text distributions 
# WITHOUT headers stripping (WITH headers)
named_target = map(lambda x: tr.target_names[x]  , tr.target)

text_lengths = pd.DataFrame({
        "lengths": list(map(len, tr['data'])),
        "label" : list(named_target)
        })

print(text_lengths.describe())

#%%
import seaborn as sns

sns.set(style="whitegrid", rc={'figure.figsize':(16.0,10.0)})
ax = sns.distplot(text_lengths['lengths'], hist=False, rug=True);

ax.set_title("20 Newsgrous - Doucment length distribution") 

#%%
import seaborn as sns

sns.set(style="whitegrid", rc={'figure.figsize':(16.0,2.0)})
ax = sns.boxplot(x='lengths', data=text_lengths);

ax.set_title("20 Newsgrous - Doucment length distribution") 

#%%

import seaborn as sns

sns.set(style="whitegrid", rc={'figure.figsize':(16.0,10.0)})
ax = sns.boxplot(x="lengths", y="label", color="grey", data=text_lengths)
ax.set_title("20 Newsgrous - Length of documents by class") 


#%%

# Subject extraction from e-mail instances

def subject(text):
    """ Extracts subject from e-mail text
    """
    REGEX = r'Subject: *(.*) *$'
    regex = re.compile(REGEX, re.MULTILINE)
    return regex.findall(text)[0]


#%%

# Cardinality info (subjects)
    
print( raw_data_train.keys() )
print( raw_data_test.keys() )


s1 = raw_data_train.data[3]


subjects_train = pd.Series( map(subject, raw_data_train.data) )
subjects_test = pd.Series( map(subject, raw_data_test.data) )

cardinality_info_train = subjects_train.describe()
print(cardinality_info_train)

#%% 

# Number of words info

count_vect = CountVectorizer()
kk = count_vect.fit_transform(subjects_train)

analyze = count_vect.build_analyzer()

subjects_words_count = subjects_train.apply(lambda x: len(analyze(x)))

print(subjects_words_count.describe())

#%%
import ggplot as gg

df = pd.DataFrame(subjects_words_count, columns = ["count"])

hist =  gg.ggplot(df, gg.aes(x = "count"))
hist += gg.xlab("# of words") +\
        gg.ylab("Frequency") +\
        gg.ggtitle("Frequency of words")

hist += gg.geom_vline(x = df.mean(), color="red")
hist += gg.geom_vline(x = df.median(), color="blue")
hist += gg.geom_density(color="green")
hist += gg.geom_histogram(binwidth=1, color="grey")

hist

#%%

# 1st attemtp to classify subjects per tag

X_raw_train = subjects_train
X_raw_test = subjects_test

Y_train = raw_data_train.target
Y_test = raw_data_test.target

target_names = raw_data_train.target_names
def get_target_name(index):
    return target_names[index]

#
# Count Vectorizer
#

count_vect = CountVectorizer()

X_train = count_vect.fit_transform(X_raw_train)
X_train_array = X_train.toarray()
X_test = count_vect.transform(X_raw_test)
X_test_array = X_test.toarray()


#
# TF-IDF
#
tfidf = TfidfTransformer(norm='l1')
X_tfidf_train = tfidf.fit_transform(X_train)
X_tfidf_test = tfidf.transform(X_test)

#%%

#
# Learn with Naive Bayes
#

# Choose multinomial assumption 

classifier_bayes = MultinomialNB()

classifier_bayes.fit(X_tfidf_train, Y_train)
score_MultinomialNB = classifier_bayes.score(X_tfidf_test, Y_test)
print( f"Score MultinomialNB : {score_MultinomialNB}")

classifier_bayes.class_prior
#%%

#
# Learn with LinearSVM
#

classifier_linearsvm = LinearSVC()
classifier_linearsvm.fit(X_tfidf_train, Y_train)
score_LinearSVC = classifier_linearsvm.score(X_tfidf_test, Y_test)
print( f"Score LinearSVC : {score_LinearSVC}" )

#%%

#
# Learn with Logistic Regression
#

classifier_log = LogisticRegression()
classifier_log.fit(X_tfidf_train, Y_train)
score_Log = classifier_log.score(X_tfidf_test, Y_test)
print( f"Score Log : {score_Log}" )


#%%

#
# Learn with decission tree classifier
#

classifier_tree = DecisionTreeClassifier()
classifier_tree.fit(X_tfidf_train, Y_train)
score_Tree = classifier_tree.score(X_tfidf_test, Y_test)
print( f"Score Tree : {score_Tree}" )

#%%

#
# Learn with random forest classifier
#

classifier_forest = RandomForestClassifier(n_estimators=100)
classifier_forest.fit(X_tfidf_train, Y_train)
score_Forest = classifier_forest.score(X_tfidf_test, Y_test)
print( f"Score Forest : {score_Forest}" )

#%%

#
# Learn with gradient boost classifier
#

classifier_gradientboost = GradientBoostingClassifier()
classifier_gradientboost.fit(X_tfidf_train, Y_train)
score_gradientboost = classifier_gradientboost.score(X_tfidf_test, Y_test)
print( f"Score Gradient Boost : {score_gradientboost}" )


#%%

#
# Predict
#

def predict(sentence, classifier):
    
    # Convert sentence to tfidf
    count_vector = count_vect.transform([sentence])
    tfidf_vector = tfidf.transform(count_vector)
    prediction = classifier.predict(tfidf_vector)[0]
    return get_target_name(prediction)


    
predict("top gear", classifier_linearsvm)

def predict_from_test_instances(idx,classifier):
    subj =  X_raw_test[idx]
    print(subj)
    prediction = predict(subj, classifier)
    print(prediction)
    tag = get_target_name(Y_test[idx])
    print(tag)
    print()
    

for i in range(10,20): 
    predict_from_test_instances(i, classifier_linearsvm)

#%%

p = X_raw_test[13]

sum( X_raw_train == p )

#
# Some subjects are repeated between train and test set.
# We should take them out!!!
# Easier by checking stirng "Re:"
#

#%%

#
# Keras: Embeddings + CNN
#

#%%
#
# CNNs - with Keras
#

#
# REF: Chollet
#

import os

from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

import pandas as pd

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, MaxPool1D, Dropout
from keras import layers, Input, Model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.constraints import max_norm

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
#%%
#
# DEFINITIONS
#

MAX_WORDS = 10000
MAX_LENGTH = 1000
EMBEDDING_DIM = 100

CNN_FILTERS_L1 = 100 # number of feature maps as per paper Zhang et al.
CNN_LENGTH_1 = 3
#CNN_LENGTH_1 = 1
CNN_LENGTH_2 = 5
CNN_LENGTH_3 = 7

N_CATEGORIES = 20

EMBEDDING_PRELOAD = False
EMBEDDING_TRAIN = True

#%%

# Import raw data

raw_data_train = fetch_20newsgroups(subset='train', random_state = 42)
raw_data_test = fetch_20newsgroups(subset='test', random_state = 42)

#%%

# Labels
labels = raw_data_train.target_names

#%%

texts_train = raw_data_train.data
texts_test  = raw_data_test.data  

labels_train = raw_data_train.target
labels_test  = raw_data_test.target

#%%
#
# Length stats of texts (in chars)
#

pd.Series(texts_train).apply(len).describe()


#%%
#
# Tokenize
#
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts_train)
print(f'Found {len(tokenizer.word_index)} different words but we limit them to the top {MAX_WORDS}.')

texts_sequences_train = tokenizer.texts_to_sequences(texts_train)
texts_sequences_test  = tokenizer.texts_to_sequences(texts_test)

#%%
#
# Reverse token index
reversed_word_index = dict( zip( tokenizer.word_index.values(),
                                 tokenizer.word_index.keys()))
    


#%%
#
# Lenght stats in number of words
#
print( f"Lengths in words:\n {pd.Series(texts_sequences_train).apply(len).describe()}")
pd.Series(texts_sequences_train).apply(len).quantile(q=.95)

#%%
#
# Pad sequencies and generating input matrix
#

x_train = pad_sequences(texts_sequences_train, maxlen=MAX_LENGTH)
x_test  = pad_sequences(texts_sequences_test,  maxlen=MAX_LENGTH)

y_train = to_categorical( np.asarray(labels_train), N_CATEGORIES)
y_test  = to_categorical( np.asarray(labels_test), N_CATEGORIES)

#%%
import numpy as np
#
# Load Glove embeddings
#
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


#%%
#
# Model
#

model = Sequential()

# Embedding not pre-trained. TODO: try with Glove
model.add( Embedding(MAX_WORDS, EMBEDDING_DIM, input_length = MAX_LENGTH) )

# Preload embeddings (Glove)
if EMBEDDING_PRELOAD:
    model.layers[0].set_weights( [embedding_matrix] )
# Do we have to train embedding layer?
model.layers[0].trainable = EMBEDDING_TRAIN

# CNN
model.add( Conv1D(CNN_FILTERS_L1, CNN_LENGTH_1, activation='relu', padding='same') )
#model.add( MaxPool1D(10) )
#model.add( Conv1D(CNN_FILTERS_L1, CNN_LENGTH_1, activation='relu', padding='same') )
model.add( GlobalMaxPool1D() )

# Dense
model.add( Dropout(0.5) )
model.add( Dense(20, activation='softmax', kernel_constraint=max_norm(3.0)) )


model.summary()

#%%

# Viz model

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='/Users/alvaro/Downloads/model.png')

#%%

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

#%%
#
# Compile and train
#
# TODO: change optimizer from RMSPROP to SGD as per paper?
model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['acc']
              )

history = model.fit(x_train, y_train,
                    epochs= 20,
                    batch_size = 128,
                    validation_split = 0.2
                    )

#%%
accuracies =  history.history['acc']
accuracies_val = history.history['val_acc']
indexes = range(len(accuracies))

plt.plot(indexes, accuracies, "bo", label="Training")
plt.plot(indexes, accuracies_val, "b", label="Validation")
plt.title("Accuracy - Epoch")
plt.legend()

#%%

# plot with ggplot

import ggplot as gg

# ggplot needs data to be in Pandas

data = pd.DataFrame(
        {"train": accuracies, 
         "validation": accuracies_val,
         "epoch": range(len(accuracies)),
         })
data_melted = data.melt(id_vars="epoch")

p = gg.ggplot(data_melted, gg.aes(x="epoch", y="value", color="variable"))
p = p + gg.geom_point() + gg.geom_line()
p

#%%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

#%%
# Evaluate model (last epoch)
accuracy_FINAL = model.evaluate(x_test, y_test)[1]

#%%

# Test accuracy with sklearn

y_predicted = model.predict(x_test)
y_predicted_argmax = np.asarray( y_predicted.argmax(axis=-1) )

#%%

y_predicted_discreet = np.zeros(y_predicted.shape)
for i,j in enumerate(y_predicted_argmax):
    y_predicted_discreet[i][j] = 1

accuracy_FINAL2 = accuracy_score(y_test, y_predicted_discreet)

# % deviation from Keras to sklearn
print(f"% error in accuracy between Keras and Sklearn: {(accuracy_FINAL2-accuracy_FINAL)/(accuracy_FINAL)*100} %")

#%%

# Confusion Matrix

cmatrix = confusion_matrix(y_test.argmax(axis=-1), y_predicted_discreet.argmax(axis=-1))

df = pd.DataFrame(cmatrix, index = labels, columns=labels)

import seaborn as sns
sns.heatmap(df, cmap='Greys')

#%%

# With ggplot
import ggplot as gg

df_ = df.copy()
df_["cat"] = df_.index
df_melted = df_.melt(id_vars=["cat"])

cm2 = gg.ggplot(df_melted, gg.aes(x="cat", fill="variable", y ="value"))

cm2 += gg.xlab("category") + gg.ylab("frequency") +\
       gg.ggtitle("Confusion Matrix")
       


cm2 += gg.geom_bar(stat="identity", position="stack")

cm2

#%%

# With altair

import altair as alt


chart = alt.Chart(df_melted).mark_bar().encode(
    x='cat',
    y='value',
    color='variable'
)

print(chart.to_json())

with open("/tmp/chart", "w") as f:
    f.write(chart.to_json())

#%%

# Classification report

report = classification_report(y_test, y_predicted_discreet)

print(f"Classification report:\n\n{report}")

average_precision_score(y_test, y_predicted_discreet)

# 

#%%
# Predict

def predict(text):
    """ Implicit dependencies
        - tokenizer
        - MAX_LENGTH
        - model
        - labels
        
        Lib dependencies
        - pad_sequences
    """
    
    print(f'Predicting text: "{text}"')
    
    # Tokenize and convert to sequence
    sequences = tokenizer.texts_to_sequences([text])
    sequences_matrix = pad_sequences(sequences, maxlen= MAX_LENGTH)
    
    # Predict
    predicted = model.predict(sequences_matrix)[0]
    
    # Most probable label
    best_label = labels[predicted.argmax()]
    
    
    return best_label
    

#%%
predict("Which car is faster?"), predict(texts_test[9])

#%%
# Model 2 - Graph
#

input_tensor = Input(shape=(MAX_LENGTH,))
embedding_layer = layers.Embedding(MAX_WORDS, EMBEDDING_DIM)
embedding_layer.trainable = EMBEDDING_TRAIN
x = embedding_layer(input_tensor)
#x = layers.Dropout(0.5)(x)

# Branch a
a = layers.Conv1D(CNN_FILTERS_L1, CNN_LENGTH_1, activation='relu', padding='same')(x)
a = layers.GlobalMaxPool1D()(a)
# Branch b
b = layers.Conv1D(CNN_FILTERS_L1, CNN_LENGTH_2, activation='relu', padding='same')(x)
b = layers.GlobalMaxPool1D()(b)
# Branch c
c = layers.Conv1D(CNN_FILTERS_L1, CNN_LENGTH_3, activation='relu', padding='same')(x)
c = layers.GlobalMaxPool1D()(c)
# Concatenate a and b and c
y = layers.concatenate([a,b,c], axis=-1)
# Dense output layer
output_tensor = layers.Dense(20, activation='softmax')(y)

model2 = Model(input_tensor, output_tensor)

model2.summary()

#%%

# Viz model

from keras.utils.vis_utils import plot_model
plot_model(model2, to_file='/Users/alvaro/Downloads/model2.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model2).create(prog='dot', format='svg'))

#%%
#
# Compile and train
#
model2.compile(optimizer='rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['acc']
               )

history2 = model2.fit(x_train, y_train,
                      epochs= 20,
                      batch_size = 128,
                      validation_split = 0.2
                      )

#%%
accuracies2 =  history2.history['acc']
accuracies_val2 = history2.history['val_acc']
indexes2 = range(len(accuracies2))

plt.plot(indexes2, accuracies2, "bo", label="Training")
plt.plot(indexes2, accuracies_val2, "b", label="Validation")
plt.title("Accuracy - Epoch")
plt.legend()

#%%
#
# TODO: shuffle?
#





