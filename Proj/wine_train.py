## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# clf = RandomForestClassifier(n_estimators=10, random_state=20701)

import os

import numpy
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.initializers.initializers_v1 import TruncatedNormal
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from numpy import argmax
from sklearn import preprocessing, model_selection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

OPTIMIZER = 'adam'
# LOSS = 'categorical_crossentropy'
LOSS = 'binary_crossentropy'
EPOCHS = 800

# from kt_utils import *

# Read data from csv default separator is ',', but file uses ';'
cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "quality"]
data = pd.read_csv("winequality-red.csv", sep=";")

data["quality"] = data["quality"].astype(int)
# data = pd.get_dummies(data, columns=["quality"])
# print(data.head(5))

# X is data columns
# X = data.iloc[:, 0:11].values
## Y is quality
# Y = data.iloc[:, 11:].values
# Data to [0,1]


X = data.iloc[:, 0:11].values
X = preprocessing.normalize(X, axis=0)
Y = np.ravel(data.quality)
# Y = to_categorical(Y, num_classes=10)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,
                                                                    Y,
                                                                    test_size=0.33,
                                                                    random_state=42)

# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, )


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# rint(X_train)
# rint(X_test)
# rint(Y_train)
# rint(Y_test)
# input()

# inputs = tf.keras.Input(shape=(shape_x,))
# x = tf.keras.layers.Dense(200, activation='linear')(inputs)
# outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)


model = Sequential()
init = TruncatedNormal(stddev=0.01, seed=10)

# config model
# model.add(Input(shape=(11,)))
model = Sequential()
model.add(Dense(164, input_dim=11, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
print(model.summary())
# input()

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['mse'])

checkpoint = ModelCheckpoint(
    filepath='BEST_' + "wine_model.h5",
    monitor='mse',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto', save_freq='epoch')

# fit
history = model.fit(x=X_train, y=Y_train, batch_size=8, epochs=EPOCHS,
                    validation_data=(X_test, Y_test), callbacks=[checkpoint])

print("Saving model and plots...")
model.save("wine_model.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(range(1, EPOCHS + 1), history.history["loss"], label="train loss")
plt.plot(range(1, EPOCHS + 1), history.history["val_loss"],
         label="validation loss")
plt.plot(range(1, EPOCHS + 1), history.history["mse"],
         label="train mse")
plt.plot(range(1, EPOCHS + 1), history.history["val_mse"],
         label="validation mse")
plt.title("Train/validation loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.ylim([0,1])
plt.savefig("wine.png")

# print("Loss = ", str(preds[0]))
# print("Test Accuracy = ", str(preds[1]))
#
# print(X_train[3])
# print(Y_train[3])
# print(model.predict(X_train[3]))

# TODO: Find better classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# clf = RandomForestClassifier(n_estimators=100, random_state=20701)
# clf.fit(X_train, Y_train)
#
# pred = clf.predict(X_test)
#
# for i in range(0, len(Y_test)):
#    print("Actual : {}".format(argmax(Y_test[i])), end=" - ")
#    print("Predicted: ", argmax(pred[i]))
#
# hit = 0
# miss = 0
# for i in range(0, len(Y_test)):
#   actual_class = Y_test[i]
#   pred_class = pred[i]
#   if actual_class != pred_class:
#       miss += 1
#       print("Actual: {}".format(actual_class), end=" - ")
#       print("Predicted: {}".format(pred_class))
#   else:
#       hit += 1

# print("\nGood predictions : {}".format(hit))
# print("Bad predictions : {}".format(miss))
# print("Total accuracy: \t", clf.score(X_test, Y_test))
