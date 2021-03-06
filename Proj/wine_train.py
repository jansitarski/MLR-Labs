import argparse
import os
import time

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.version_utils import callbacks
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(20701)
EPOCHS = 100
OPTIMIZER = 'adam'
LOSS = "categorical_crossentropy"
# Hinders accuracy to 60%, Y = to_categorical(Y, num_classes=10)
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--optimizer", required=True, help="Optimizer to use")
ap.add_argument("-e", "--epochs", required=True, help="Number of epochs")
args = vars(ap.parse_args())

EPOCHS = int(args["epochs"])
OPTIMIZER = args["optimizer"]

data = pd.read_csv("winequality-red.csv", sep=";")

# Create columns for each quality [1-10],
# Note: current dataset has only 6 qualities
data = pd.get_dummies(data, columns=["quality"])

X = data.iloc[:, 0:11].values
Y = data.iloc[:, 11:].values

for index, dataset in enumerate(X):
    if np.argmax(dataset) > 1000:
        np.delete(X, index)
        np.delete(Y, index)

numberOfQualities = len(Y[0])

# Standardize X values
scaler = StandardScaler()
scaler = scaler.fit(X)
X[:] = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(11,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(numberOfQualities, activation='softmax'))

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    filepath='BEST_' + "wine_model.h5",
    monitor='val_accuracy',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto', save_freq='epoch')


class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_callback = TimeHistory()
history = model.fit(X_train, Y_train, batch_size=8, epochs=EPOCHS, verbose=0,
                    validation_data=(X_test, Y_test),
                    callbacks=[checkpoint, time_callback])

times = time_callback.times
print("dt:", str(sum(times)))
# print(history.history["val_loss"])
print("result: ", end="")
for i in range(0, EPOCHS):
    print(str(history.history["val_accuracy"][i]),end=" ")
print("\n")

print("Saving model and plots...")
model.save("wine_model.h5")

plt.style.use("ggplot")
plt.figure()
plt.ylim([0.2, 2])
plt.plot(range(1, EPOCHS + 1), history.history["loss"], label="train loss")
plt.plot(range(1, EPOCHS + 1), history.history["val_loss"],
         label="validation loss")
plt.plot(range(1, EPOCHS + 1), history.history["accuracy"],
         label="train mae")
plt.plot(range(1, EPOCHS + 1), history.history["val_accuracy"],
         label="validation mse")
plt.title("Train/validation loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Metrics")
plt.legend()
plt.savefig("model.png")
