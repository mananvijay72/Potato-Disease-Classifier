import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import io
import os

# Load the pre-trained model
model = load_model(r'models\model.keras')

# Define the class labels
class_labels = os.listdir('data')

def preprocess_image(image):
    # Convert to RGB
    image = image.convert('RGB')
    # Resize image to match model input
    image = tf.image.resize(image, (256,256))

    # Convert image to numpy array
    image_array = np.array(image) / 255.0  # Normalize
    # Expand dimensions to match model input
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def prediction(image_array):
    # Predict the class
    predictions = model.predict(image_array)
    # Get the class with the highest probability
    index = np.argmax(predictions.reshape(-1))
    class_labels = os.listdir('data')
    return class_labels[index]

# Streamlit app
st.title("Emotion Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Process the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image_array = preprocess_image(image)
    
    # Predict the emotion
    prediction = prediction(image_array)
    
    # Display the result
    st.write(f"Prediction: {prediction}")

# Webcam input
st.write("Or use your webcam to capture an image:")

# Create a video capture object
video_capture = cv2.VideoCapture(0)

# Capture image from webcam
if st.button('Capture Image'):
    ret, frame = video_capture.read()
    if ret:
        # Convert the frame to an image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(image, caption='Captured Image', use_column_width=True)
        
        # Preprocess the image
        image_array = preprocess_image(image)
        
        # Predict the emotion
        prediction = prediction(image_array)
        
        # Display the result
        st.write(f"Prediction: {prediction}")

# Release the webcam capture object
video_capture.release()

