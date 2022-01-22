# python classify.py -h
#
# Optionally, prepare text JSON file 'test_labels.txt' with the actual classes of the test images:
# {"filename1" : "class1", "filename2" : "class2", ... }
# If that file is found in the same directory as images, JSON will be used as dictionary of labels.

# Import packages
import os

# Disable TF messages and CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import argparse
import cv2
import pickle
import json

false_positive = 0
true_positive = 0
false_negative = 0
true_negative = 0
best_false_positive = 0
best_true_positive = 0
best_false_negative = 0
best_true_negative = 0

# Size/resize parameters (they should be the same as used in training)
SIZE = 128
DO_RESIZE = True

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="Trained model file name (without extension)")
ap.add_argument("-t", "--testset", required=True, help="Path to test images")
ap.add_argument("-c", "--console", required=False, action='store_true',
                help="display results in console only", default=False)
args = vars(ap.parse_args())

console_only = args["console"]

# Read labels for classes to recognize
print("Loading labels...")
with open(args["model"] + ".lbl", 'rb') as f:
    CLASS_LABELS = pickle.load(f)

# Load test labels from file, if exists
try:
    with open(args["testset"] + os.path.sep + "test_labels.txt") as f:
        data = f.read()
    test_labels = json.loads(data)
except FileNotFoundError:
    test_labels = False

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(args["model"] + ".h5")
if os.path.isfile('./BEST_' + args["model"] + ".h5"):
    bestModel = tf.keras.models.load_model('BEST_' + args["model"] + ".h5")

# Loop over images
print("Classifying...")

for image_name in sorted(os.listdir(args["testset"])):

    # Skip to the next file if not an image
    # This could be avoided by using imutils.paths.list_images(), like in the 'train.py'
    if not image_name.endswith((".jpg", ".png")): continue

    # Load the image
    image = cv2.imread(args["testset"] + os.path.sep + image_name)

    # Pre-process the image for classification
    if DO_RESIZE:
        image_data = cv2.resize(image, (SIZE, SIZE))
    image_data = image_data.astype("float") / 255.0
    image_data = tf.keras.preprocessing.image.img_to_array(image_data)
    image_data = np.expand_dims(image_data, axis=0)

    # Classify the input image
    prediction = model.predict(image_data)

    if os.path.isfile('./BEST_' + args["model"] + ".h5"):
        bestPrediction = bestModel.predict(image_data)
        # Find and print the winner class and the probability
        bestWinner_class = np.argmax(bestPrediction)
        bestWinner_probability = np.max(bestPrediction) * 100

    # Find and print the winner class and the probability
    winner_class = np.argmax(prediction)
    winner_probability = np.max(prediction) * 100

    print("File: {}, prediction - {}: {:.2f}%".format(image_name,
                                                      CLASS_LABELS[
                                                          winner_class],
                                                      winner_probability),
          end="")

    # Print actual class, if available
    if test_labels:
        actual_class = test_labels[image_name]

        print("; actual class - {} \t {}".format(actual_class,
                                                 "OK"
                                                 if actual_class ==
                                                    CLASS_LABELS[winner_class]
                                                 else "Error!"))
        #print(CLASS_LABELS[0]) #OTHER
        #print(CLASS_LABELS[1]) #ZEBRAS
        try:
            CLASS_LABELS[2]
        except IndexError:
            #print("winner class: ", winner_class)
            #print("accual class: ", actual_class)
            #print("CLASS LABEL WINNR class: ", CLASS_LABELS[winner_class])
            if actual_class == CLASS_LABELS[winner_class] and actual_class == CLASS_LABELS[1]:
                print("true_positive")
                true_positive+=1
            elif actual_class != CLASS_LABELS[winner_class] and actual_class == CLASS_LABELS[1]:
                print("false_negative")
                false_negative+=1
            elif actual_class == CLASS_LABELS[winner_class] and actual_class == CLASS_LABELS[0]:
                print("true_negative")
                true_negative+=1
            elif actual_class != CLASS_LABELS[winner_class] and actual_class == CLASS_LABELS[0]:
                print("false_positive")
                false_positive+=1

    else:
        print()

    if os.path.isfile('./BEST_' + args["model"] + ".h5"):
        print("File: {}, Best prediction - {}: {:.2f}%".format(image_name,
                                                               CLASS_LABELS[
                                                                   bestWinner_class],
                                                               bestWinner_probability),
              end="")

    # Print actual class, if available
    if test_labels:
        actual_class = test_labels[image_name]
        if os.path.isfile('./BEST_' + args["model"] + ".h5"):
            print("; actual class - {} \t {}".format(actual_class,
                                                     "OK"
                                                     if actual_class ==
                                                        CLASS_LABELS[
                                                            bestWinner_class]
                                                     else "Error!"))


        try:
            CLASS_LABELS[2]
        except IndexError:
            if actual_class == CLASS_LABELS[bestWinner_class] and actual_class == CLASS_LABELS[1]:
                print("true_positive")
                best_true_positive+=1
            elif actual_class != CLASS_LABELS[bestWinner_class] and actual_class == CLASS_LABELS[1]:
                print("false_negative")
                best_false_negative+=1
            elif actual_class == CLASS_LABELS[bestWinner_class] and actual_class == CLASS_LABELS[0]:
                print("true_negative")
                best_true_negative+=1
            elif actual_class != CLASS_LABELS[bestWinner_class] and actual_class == CLASS_LABELS[0]:
                print("false_positive")
                best_false_positive+=1

        print()
    else:
        print()

    if not console_only:
        # Build the label
        label = "{}: {:.2f}%".format(CLASS_LABELS[winner_class],
                                     winner_probability)

        if os.path.isfile('./BEST_' + args["model"] + ".h5"):
            label = "{}: {:.2f}%, Best: {}: {:.2f}%".format(
                CLASS_LABELS[winner_class],
                winner_probability, CLASS_LABELS[bestWinner_class],
                bestWinner_probability)

        # Draw the label on the image
        output = cv2.resize(image, (600, 600))
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # Show the output image
        cv2.imshow("Output", output)
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Break on 'q' pressed
            break

print("True Positive: ", true_positive)
print("True Negative: ", true_negative)
print("False Positive: ", false_positive)
print("False Negative: ", false_negative)
if os.path.isfile('./BEST_' + args["model"] + ".h5"):
    print("\nBest True Positive: ", best_true_positive)
    print("Best True Negative: ", best_true_negative)
    print("Best False Positive: ", best_false_positive)
    print("Best False Negative: ", best_false_negative)
# Zadanie 12.3
# W przypadku, gdy oprócz pliku z modelem dla ostatniej epoki treningu istnieje plik 
# z najlepszym modelem 'best' (o nazwie args["model"]+"-best.h5", utworzonym przez callback 
# ModelCheckpoint przez 'train.py'), skrypt powinien dokonywać dodatkowej predykcji dla tego modelu
# a wyniki predykcji wyświetlać w konsoli w postaci np.:
# File: 1.jpg, prediction - cats: 99.93%; actual class - dogs      Error!
# (Best model) File: 1.jpg, prediction - dogs: 98.92%; actual class - dogs      OK
# oraz na etykiecie w obrazie, w formie np.:
# cats: 99.93%; (Best model) dogs: 98.92%

# Zadanie 12.4
# Dla swojego zbioru testowego przygotować plik 'test_labels.txt', na wzór tego zamieszczonego 
# w przykładzie 'cats_dogs-1000'.
# Uzupełnić program o obliczanie i wypisywanie macierzy/tablicy pomyłek,
# https://pl.wikipedia.org/wiki/Tablica_pomy%C5%82ek  (lub wykład 06).
# Macierz powinna być wyświetlana wyłącznie w przypadku gdy liczba klas wynosi 2. W przypadku  
# gdy istnieje model 'best', to macierz powinna być wyświetlona również dla niego.
