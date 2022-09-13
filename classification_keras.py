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
# data = pd.read_csv("pdb_filtered_top10_data.csv.zip", compression='zip')
data = pd.read_csv("pdb_filtered_top5_data.csv.zip", compression='zip')

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
print("X", X.shape)
print("Y", Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.7, random_state=0)

embedding_dim = 25 # orig 8

# create the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))
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