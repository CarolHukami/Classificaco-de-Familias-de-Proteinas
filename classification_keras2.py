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
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import LabelBinarizer

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
max_length = 1200
#create and fit tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(seqs)
#represent input data as word rank number sequences
X = tokenizer.texts_to_sequences(seqs)
#X = sequence.pad_sequences(X, maxlen=max_length)
X = pad_sequences(X, maxlen=max_length)
X = np.array(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.7, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from keras.layers import Conv1D, MaxPooling1D, Concatenate, Input
from keras.models import Sequential,Model

embedding_dim = 25
units = 256
num_filters = 32
filter_sizes=(3,5, 9,15,21)
conv_blocks = []

embedding_layer = Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length)
es2 = EarlyStopping(monitor='val_acc', verbose=1, patience=4)

sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

z = Dropout(0.1)(embedded_sequences)

for sz in filter_sizes:
    conv = Conv1D(
        filters=num_filters,
        kernel_size=sz,
        padding="valid",
        activation="relu",
        strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(0.25)(z)
z = BatchNormalization()(z)
z = Dense(units, activation="relu")(z)
z = BatchNormalization()(z)
predictions = Dense(top_classes, activation="softmax")(z)
model2 = Model(sequence_input, predictions)
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())

model2.fit(X_train, y_train,  batch_size=64, verbose=1, validation_split=0.5,epochs=30) #callbacks=[es]

y_pred = model2.predict(X_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred, target_names=lb.classes_))
print("ACC:", accuracy_score(y_test, y_pred))
print("F1S:", f1_score(y_test, y_pred, average="weighted"))
print("REC:", recall_score(y_test, y_pred, average="weighted"))
print("PRE:", precision_score(y_test, y_pred, average="weighted"))