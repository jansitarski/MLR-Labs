import os

import pandas as pd
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv("winequality-red.csv", sep=";")
data = pd.get_dummies(data, columns=["quality"])

X = data.iloc[:, 0:11].values
Y = data.iloc[:, 11:].values

numberOfQualities = len(Y[0])

scaler = StandardScaler()
scaler = scaler.fit(X)
X[:] = scaler.transform(X)

_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.9)

model = keras.models.load_model("BEST_wine_model.h5")
predictions = model.predict(X_test)
hit = 0
miss = 0
nearMiss = 0
for i in range(1, len(predictions)):
    prediction = predictions[i]
    label = argmax(Y_test[i])
    if argmax(prediction) == label:
        hit += 1
    elif abs(argmax(prediction) - label) == 1:
        nearMiss += 1
    else:
        miss += 1
        print(argmax(predictions[i]), " MISS ", label)
print("Hit = ", str(hit))
print("Near Miss = ", str(nearMiss))
print("Miss = ", str(miss))
