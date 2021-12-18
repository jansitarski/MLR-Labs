import os

# Disable TF warning messages

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2  # pip install opencv-python

# Directory with test set
TEST_DATASET_DIR = 'mnist-test'

# Trained model name
MODEL = 'model.h5'

# Load trained model of neural network
model = tf.keras.models.load_model(MODEL)

# Load the images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

for image_index in range(90, 100):

    # Test image
    test_image = x_test[image_index]
    test_image = test_image.astype('float32') / 255.0

    # Classify the input image
    prediction = model.predict(test_image.reshape(1, 28, 28, 1))

    # Find the winner class and the probability
    sorted_classes = np.argsort(-1 * prediction)
    sorted_prediction = -np.sort(-1 * prediction)
    winner_probability = np.max(sorted_prediction[0][0:] * 100)
    second_probability = np.max(sorted_prediction[0][1:] * 100)
    third_probability = np.max(sorted_prediction[0][2:] * 100)

    # Build the text label
    best_label = "1st = {}, p = {:.2f}%\n".format(sorted_classes[0][0],
                                                  winner_probability)
    second_label = "2nd = {}, p = {:.2f}%\n".format(sorted_classes[0][1],
                                                    second_probability)
    third_label = "3rd = {}, p = {:.2f}%\n".format(sorted_classes[0][2],
                                                   third_probability)

    # Draw the label on the image
    output_image = cv2.resize(test_image, (500, 500))
    cv2.putText(output_image, best_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, 255, 2)
    cv2.putText(output_image, second_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, 255, 2)
    cv2.putText(output_image, third_label, (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, 255, 2)
    # Show the output image

    cv2.imshow("Output", output_image)
    cv2.waitKey(0)

# Zadanie 9.2 (1p)
# Oprócz wyświetlania na obrazach zwycięskiej klasy, wyświetlać dodatkowo
# klasę z miejsca 2 i 3 (z prawdopodobieństwami)

# Zadanie 9.3 (1.5p)
# Zamiast wczytywać obrazy testowe z pliku, wykorzystać możliwość załadowania ich
# metodą mnist.load_data()

# Wynik: plik tekstowy z uzupełnionym kodem 
#        oraz plik graficzny z przykładowym wynikiem predykcji
