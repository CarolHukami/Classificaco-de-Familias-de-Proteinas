# https://protlearn.readthedocs.io/en/latest/preprocessing.html
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


# le dataset
# data = pd.read_csv("pdb_filtered_top10_data.csv.zip", compression='zip')
data = pd.read_csv("pdb_filtered_top5_data.csv.zip", compression='zip')

# conta número de instâncias por classe
cnt = Counter(data.classification)
# mantém apenas famílias que possuem mais de 200 instâncias
classes  = {}
# ordena classes por quantia
sorted_classes = cnt.most_common()
# anda nas classes
for c in sorted_classes:
    # se tiver mais de 200, salva
    if c[1] > 200:
        # salva no dicionário
        classes[c[0]] = c[1]
for i, (classe, count) in enumerate(classes.items()):
    print(i+1, classe, '=', count)
print(data)
# pega sequencias
seqs = data.sequence.values
# pega classes
classes = data.classification.values
# transforma em lista numpy
Y = np.array(classes)
# numero maximo de features
max_length = 1200

tokenizer = CountVectorizer(analyzer='char')
X = tokenizer.fit_transform(seqs)

# X = np.array(X)
print("X", X.shape)
print("Y", Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.7, random_state=0)

# clf = KNeighborsClassifier()
# clf = LinearSVC(verbose=1)
# clf = RandomForestClassifier(n_jobs=-1)
# clf = SGDClassifier(n_jobs=-1)
# clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_jobs=-1))
clf = MLPClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("ACC:", accuracy_score(y_test, y_pred))
print("F1S:", f1_score(y_test, y_pred, average="weighted"))
print("REC:", recall_score(y_test, y_pred, average="weighted"))
print("PRE:", precision_score(y_test, y_pred, average="weighted"))