# python train.py -h
#
# Keep training images in the directories:
# <path/to/train/images>/class-0-class0name
# <path/to/train/images>/class-1-class1name
# ... etc

import os

# Disable TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# For execution time calculation
from timeit import default_timer as timer

# Import the necessary packages
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import pickle
import os
import nn

start_time = timer()

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-e", "--epochs", required=True, help="Number of epochs")
ap.add_argument("-m", "--model", required=True,
                help="Output model filename (without the extension)")
ap.add_argument("-a", "--augmentation", required=False, action='store_true',
                help="Perform augmentation")
ap.add_argument("-g", "--gpu", required=False, action='store_true',
                help="Use GPU")
args = vars(ap.parse_args())

# Collect the input data
# Random seed
RANDOM_SEED = 20701
# Batch size
BS = 32
# Size of the images (SIZE x SIZE)
SIZE = 128
# Optionally resize images to the dimension of SIZE x SIZE, if they have different sizes
DO_RESIZE = True
# Images color depth (3 for RGB, 1 for grayscale)
IMG_DEPTH = 3
# Fraction of validation data
VALID_SPLIT = 0.2
# Data taken from the command-line arguments
EPOCHS = int(args["epochs"])
DO_AUGMENTATION = args["augmentation"]
MODEL_BASENAME = args["model"]
# Prepare GPU
if args["gpu"]:
    if tf.test.is_built_with_cuda():
        gpu_devices = tf.config.list_physical_devices('GPU')
        for device in gpu_devices:
            # Limit the GPU memory to the current needs
            tf.config.experimental.set_memory_growth(device, True)
        if len(gpu_devices) < 1:
            print("Warning - no GPU available!")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Random seed, for reproductible results (works stable for CPU only)
try:
    tf.keras.utils.set_random_seed(RANDOM_SEED)  # TF 2.7.0
except AttributeError:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

# Initialize the data and labels
print("Loading images...")
data = []
labels = []
labels_text = []

# Grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.shuffle(imagePaths)

# Loop over the input images
for imagePath in imagePaths:
    # Load the image and pre-process it
    image = cv2.imread(imagePath)
    if DO_RESIZE:
        image = cv2.resize(image, (SIZE, SIZE))
        image = tf.keras.preprocessing.image.img_to_array(image)
    # Extract the class label (numeric and text) from the image path
    dirname = imagePath.split(os.path.sep)[-2]
    dirname_list = dirname.split("-")

    if dirname_list[0] != "class":
        # File not in "class-*" directory, skip to another file
        continue

    # Label is the number in the directory name, after first "-"
    label = int(dirname_list[1])
    # Text label is the text in the directory name, after second "-"
    try:
        label_text = dirname_list[2]
    except KeyError:
        label_text = int(dirname_list[1])

    # Store image and labels in lists
    data.append(image)
    labels.append(label)
    labels_text.append(label_text)

# Get the unique classes names
classes = np.unique(labels_text)

# Save the text labels to disk as pickle
with open(MODEL_BASENAME + ".lbl", "wb") as f:
    f.write(pickle.dumps(classes))
# with open("BEST_"+MODEL_BASENAME + ".lbl", "wb") as f:
#   f.write(pickle.dumps(classes))

# Convert labels to numpy array
labels = np.array(labels)

# Determine number of classes
no_classes = len(classes)

# Scale the raw pixel intensities to the [0, 1] range
data = np.array(data, dtype="float") / 255.0

# Data partitioning (only if augmentation is enabled)
if DO_AUGMENTATION:
    # Partition the data into training and validating splits
    (train_data, valid_data, train_labels, valid_labels) = train_test_split(
        data, labels,
        test_size=VALID_SPLIT, random_state=RANDOM_SEED)
else:
    # Data partitioning will be done automatically during training
    train_data = data
    train_labels = labels


# Convert the labels from integers to category vectors
train_labels = tf.keras.utils.to_categorical(train_labels,
                                             num_classes=no_classes)
if DO_AUGMENTATION:
    valid_labels = tf.keras.utils.to_categorical(valid_labels,
                                                 num_classes=no_classes)

# Data augmentation
if DO_AUGMENTATION:
    print("Perform augmentation...")
    # Construct the image generator for data augmentation
    aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,
                                                          width_shift_range=0.1,
                                                          height_shift_range=0.1,
                                                          shear_range=0.2,
                                                          zoom_range=0.2,
                                                          horizontal_flip=True,
                                                          fill_mode="nearest")

# Initialize the model
print("Compiling model...")
model = nn.SmallerVGGNet.build(width=SIZE, height=SIZE, depth=IMG_DEPTH,
                               classes=no_classes)

# LeNet is not a good choice for general image classification, since it works on 32x32 pixel images,
# use for testing only
# model = nn.LeNet5.build(width=SIZE, height=SIZE, depth=IMG_DEPTH, classes=no_classes)

# Fully Connected - also not a very good choice, use for testing only; tune the hidden_units
# model = nn.FullyConnectedForImageClassisfication.build(width=SIZE, height=SIZE, depth=IMG_DEPTH,
#                                                       hidden_units=1000, classes=no_classes)
model.summary()

# Select the loss function
if no_classes == 2:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"

# Compile model
model.compile(loss=loss, optimizer="Adam", metrics=["accuracy"])

# Zadanie 12.1
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='BEST_' + MODEL_BASENAME + ".h5",
    monitor='val_accuracy',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto', save_freq='epoch')

# Train the network
print("Training network...")
if DO_AUGMENTATION:
    H = model.fit(x=aug.flow(train_data, train_labels, batch_size=BS),
                  epochs=EPOCHS,
                  validation_data=(valid_data, valid_labels),
                  callbacks=[checkpoint])
else:
    H = model.fit(x=train_data, y=train_labels, batch_size=BS, epochs=EPOCHS,
                  validation_split=VALID_SPLIT, callbacks=[checkpoint])

# Save model to disk
print("Saving model and plots...")
model.save(MODEL_BASENAME + ".h5")

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], label="train_loss")
plt.plot(range(EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(range(EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(range(EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(MODEL_BASENAME + ".png")

# Finishing
end_time = timer() - start_time
print("... finished in {:.2f} seconds.".format(end_time))

# Zadanie 12.1
# Uzupełnić program o callback ModelCheckpoint, https://keras.io/api/callbacks/model_checkpoint/
# tak, żeby zapisywał najlepszy model ('best') w pliku o nazwie MODEL_BASENAME+"-best.h5".
# Jako kryterium najlepszego modelu przyjąć maksimum dokładności na zbiorze walidującym (val_accuracy).
#


# Zadanie 12.2
# Pobrać z Internetu po kilkaset obrazów z dwóch kategorii, jedna z nich powinna zawierać
# obrazy z danej kategorii (np. "koty"), a druga - pozostałe obrazy ("inne").
# Przykładowe oprogramowanie do pobierania obrazów: https://www.webimagedownloader.com/ 
# (20-dniowa licencja), lub dowolny inny (są też dedykowane moduły do Pythona).
# Pliki podzielić zgodnie z informacją na początku skryptu, na wzór zamieszczonej 
# przykładowej struktury 'cats_dogs-1000'. Wydzielić kilkadziesiąt obrazów jako obrazy testowe.
# Przeprowadzić trening dla 100 epok, otrzymując dwa modele - główny (dla ostatniej epoki) 
# i "najlepszy", utworzony przez callback (może się zdarzyć że będą tożsame).
# Jako RANDOM_SEED przyjąć swój numer indeksu.
# W przypadku niezadawalających wyników treningu (bardzo złe wartości val_loss, val_accuracy)
# dokonać zmian w hiperparametrach, np. włączyć augmentację ("-a"), zmienić optymalizator itp.
