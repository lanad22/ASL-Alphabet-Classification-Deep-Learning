import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the model
model = load_model('vgg_1_asl_alphabet.h5')

# Constants
IMAGE_SIZE = 200
CROP_SIZE = 400
TRAINING_PATH = 'asl_alphabet_train/'

classes = os.listdir(TRAINING_PATH)
classes.sort()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Calculate the starting points for the central crop
    x_start = (width - CROP_SIZE) // 2
    y_start = (height - CROP_SIZE) // 2

    
    cv2.rectangle(frame, (x_start, y_start), (x_start + CROP_SIZE, y_start + CROP_SIZE), (0, 255, 0), 3)
    cropped_image = frame[y_start:y_start + CROP_SIZE, x_start:x_start + CROP_SIZE]
 
    resized_image = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    image_array = img_to_array(resized_image) / 255.0  # Normalize the image
    image_batch = np.expand_dims(image_array, axis=0)  # Create a batch

    
    predictions = model.predict(image_batch)
    predicted_class = classes[np.argmax(predictions)]
    prediction_probability = np.max(predictions)


    text = f'{predicted_class} - {prediction_probability * 100:.2f}%'
    cv2.putText(frame, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

  
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
