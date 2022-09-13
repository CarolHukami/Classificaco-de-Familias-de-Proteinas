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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.neighbors import KNeighborsClassifier


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
# data = pd.read_csv("pdb_filtered_top10_data.csv.zip", compression='zip')
data = pd.read_csv("pdb_filtered_top5_data.csv.zip", compression='zip')
# pega sequencias
seqs = data.sequence.values
seqs_list = []
for s in seqs:
    seqs_list.append(list(s))
    # seqs_list.append(' '.join(list(s)))
seqs = seqs_list
# pega classes
classes = data.classification.values
# transforma em lista numpy
Y = np.array(classes)
# numero maximo de features
max_length = 1200

X = np.array(seqs)
print("X", X.shape)
print("Y", Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.7, random_state=0)

w2v = Word2Vec(X_train, vector_size=500,min_count=1, window=1200)
# get generated space
space = dict(zip(w2v.wv.index_to_key,w2v.wv.vectors))
# initialize vectorizer
m = MeanEmbeddingVectorizer(space)
m.fit(X_train)
# transform train and test texts to w2v mean
X_train = m.transform(X_train)
X_test = m.transform(X_test)

# clf = KNeighborsClassifier()
# clf = LinearSVC()
clf = RandomForestClassifier(n_jobs=-1)
# clf = SGDClassifier(n_jobs=-1)
# clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_jobs=-1))
# clf = MLPClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("ACC:", accuracy_score(y_test, y_pred))
print("F1S:", f1_score(y_test, y_pred, average="weighted"))
print("REC:", recall_score(y_test, y_pred, average="weighted"))
print("PRE:", precision_score(y_test, y_pred, average="weighted"))