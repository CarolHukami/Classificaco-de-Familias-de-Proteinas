# https://www.kaggle.com/code/danofer/deep-protein-sequence-family-classification/notebook
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from gensim.models import Word2Vec

# vectorizer class: calc average of words using word2vec
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

# le dataset
data = pd.read_csv("pdb_filtered_top10_data.csv.zip", compression='zip')
# data = pd.read_csv("pdb_filtered_top7_data.csv.zip", compression='zip')

print(data)
# pega sequencias
seqs = data.sequence.values
# pega classes
classes = data.classification.values
# numero de classes
top_classes = len(list(set(data.classification.values)))
# transforma em lista numpy
Y = np.array(classes)
# Transform labels to one-hot
lb = LabelBinarizer()
Y = lb.fit_transform(Y)
# numero maximo de features
max_length = 500
X = seqs

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.7, random_state=0)

# initialize and train model
w2v = Word2Vec(X_train,vector_size=max_length,min_count=1)
# get generated space
space = dict(zip(w2v.wv.index_to_key,w2v.wv.vectors))
# initialize vectorizer
m = MeanEmbeddingVectorizer(space)
m.fit(X_train)
# transform train and test texts to w2v mean
X_train = m.transform(X_train)
X_test = m.transform(X_test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

embedding_dim = 25 # orig 8

# create the model
model = Sequential()
model.add(Embedding(max_length, embedding_dim))#, input_length=max_length))
model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu',dilation_rate=1))
# model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu')) #orig
model.add(Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu',dilation_rate=1))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=2))
model.add(AveragePooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) # orig
model.add(Conv1D(filters=128, kernel_size=7, padding='valid', activation='relu',dilation_rate=2)) 
model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=2))
model.add(AveragePooling1D(pool_size=2))

# model.add(Flatten()) ## Could do pooling instead 
# GlobalAveragePooling1D,GlobalMaxPooling1D
model.add(GlobalAveragePooling1D())

model.add(Dense(256, activation='relu')) # 128
model.add(BatchNormalization())
model.add(Dense(128, activation='relu')) # 128
model.add(BatchNormalization())
model.add(Dense(top_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=4)

model.fit(X_train, y_train,  batch_size=128, verbose=1, validation_split=0.5,epochs=25) # epochs=15, # batch_size=128 ,callbacks=[es]

y_pred = model.predict(X_test)

# print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=lb.classes_))
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred, target_names=lb.classes_))
print("ACC:", accuracy_score(y_test, y_pred))
print("F1S:", f1_score(y_test, y_pred, average="weighted"))
print("REC:", recall_score(y_test, y_pred, average="weighted"))
print("PRE:", precision_score(y_test, y_pred, average="weighted"))