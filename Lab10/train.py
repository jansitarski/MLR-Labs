import os
# Disable TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import nn
from sklearn.model_selection import train_test_split  # pip install sklearn
import matplotlib.pyplot as plt
import argparse
from timeit import default_timer as timer

# Default values (suitable for MNIST)
MODEL_FILENAME = "model"
NN_TYPE = "lenet5"
EPOCHS = 10
CLASSES = 10
BATCH_SIZE = 32
TRAIN_VALIDATION_SPLIT = 0.2
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
IMAGE_DIMENSIONS = (28, 28, 1)
DO_AUGMENTATION = False
FC_HIDDEN_UNITS = 50
# Random seed is student index s20701
RANDOM_SEED = 20701

# Construct the argument parser and parse the input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, 
                help="output model file name (extension .h5 will be added)",
                default=MODEL_FILENAME, metavar="filename")
ap.add_argument("-n", "--network", required=False, choices=['fc', 'lenet5'], 
                help="network architecture", default=NN_TYPE)
ap.add_argument("-e", "--epochs", required=False, type=int, help="number of epochs",
                default=EPOCHS, metavar='int')
ap.add_argument("-a", "--augmentation", required=False, action='store_true', 
                help="enable augmentations", default=DO_AUGMENTATION)

args = vars(ap.parse_args())
model_filename = args["model"]
nn_type = args["network"]
epochs = args["epochs"]
do_augmentation = args["augmentation"]

# Random seed, for reproductible results
tf.keras.utils.set_random_seed(RANDOM_SEED)

# Load dataset as train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Convert from uint8 to float32 and normalize to [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# Transform labels to 0-1 encoding, e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_train = tf.keras.utils.to_categorical(y_train, CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, CLASSES)

if do_augmentation:
    # Partition the data into training and validating splits
    (x_train, x_valid, y_train, y_valid) = train_test_split(
                                x_train, y_train,
                                test_size = TRAIN_VALIDATION_SPLIT,
                                random_state=RANDOM_SEED)

# Reshape the dataset into 4D array (required by Keras)
# First dimension is the number of samples
x_train = x_train.reshape(x_train.shape[0], *IMAGE_DIMENSIONS)
x_test = x_test.reshape(x_test.shape[0], *IMAGE_DIMENSIONS)
if do_augmentation:
    x_valid = x_valid.reshape(x_valid.shape[0], *IMAGE_DIMENSIONS)

# Construct the model
if nn_type == "fc":
    model = nn.FullyConnectedForMnist.build(FC_HIDDEN_UNITS)
elif nn_type == "lenet5":
    model = nn.LeNet5.build(*IMAGE_DIMENSIONS, CLASSES)

if do_augmentation:
    # Construct the image generator for data augmentation
    aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, zoom_range=0.2,
            shear_range=0.2, fill_mode="nearest")

# Compile the model and print summary
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy', 'Precision', 'Recall'])
model.summary()

# Train the model
start_time = timer()
if do_augmentation:
    # Make the real-time augmentations
    h = model.fit(aug.flow(x_train, y_train, batch_size=BATCH_SIZE),
                  epochs=epochs, steps_per_epoch=len(x_train)/BATCH_SIZE, 
                  validation_data=(x_valid, y_valid))
    # For TF <= 2.0: h = model.fit_generator(...)
else:
    h = model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
                  validation_split=TRAIN_VALIDATION_SPLIT)
    # ... or use validation_data= instead of validation_split=
training_time = timer()-start_time
print("Training time: {:.2f} s.".format(training_time))

# Save model to a file
model.save(model_filename+".h5")

# Evaluate the model
model.evaluate(x_test, y_test)

# Make plot
plt.style.use("ggplot")
plt.figure()
plt.plot(range(1,epochs+1), h.history["loss"], label="train loss")
plt.plot(range(1,epochs+1), h.history["val_loss"], label="validation loss")
plt.plot(range(1,epochs+1), h.history["accuracy"], label="train accuracy")
plt.plot(range(1,epochs+1), h.history["val_accuracy"], label="validation accuracy")
plt.title("Train/validation loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(model_filename+".png")
