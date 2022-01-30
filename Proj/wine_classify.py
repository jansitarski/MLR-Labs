import os

import numpy
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.initializers.initializers_v1 import TruncatedNormal
from keras.layers import Dense
from keras.models import Sequential, load_model
from matplotlib import pyplot as plt
from numpy import argmax
from sklearn import preprocessing, model_selection
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv("winequality-red.csv", sep=";")

data["quality"] = data["quality"].astype(int)

X = data.iloc[:, 0:11].values
X = preprocessing.normalize(X, axis=0)
Y = np.ravel(data.quality)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,
                                                                    Y,
                                                                    test_size=0.33,
                                                                    random_state=42)

model = keras.models.load_model("BEST_wine_model.h5")
hit = 0
miss = 0
for i in range(1, len(X_test)):
    data = np.expand_dims(X_test[i], axis=0)
    label = Y_test[i]
    # print(data)
    # model.predict(data)
    # print(argmax(model.predict(data)))
    if abs(numpy.round(model.predict(data)[0],decimals=1) - Y_test[i]) < 0.6:
        #print(numpy.rint(model.predict(data)[0]), " HIT ", Y_test[i])
        hit+=1
    else:
        #print(numpy.round(model.predict(data)[0],decimals=1), " MISS ", Y_test[i])
        miss+=1
    # print(Y_test[i][argmax(model.predict(data))])
print("Hit = ", str(hit))
print("Miss = ", str(miss))