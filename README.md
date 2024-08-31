# Potato Disease Classifier

## Project Overview
This project aims to classify potato leaf images into three categories: **Early Blight**, **Late Blight**, and **Healthy**. Using a Convolutional Neural Network (CNN) built with TensorFlow, the model predicts the condition of the potato leaf from the input image. The web application interface is created using FastAPI, with basic HTML and CSS for the frontend.

## Problem Statement
The goal is to predict the disease affecting a potato leaf based on its image. The model identifies whether the leaf is affected by Early Blight, Late Blight, or is Healthy.

## Project Structure
- **notebook**: Contatins the Jupyter notebook with the code for data preprocessing, model training, and evaluation.
- **models/**: Contains the saved TensorFlow model.
- **app/**: FastAPI application code.
- **data/**: Contains sample images in lable named directories for testing.

## Approach
1. **Data Collection**: Images of potato leaves were collected and labeled into three categories: Early Blight, Late Blight, and Healthy.
2. **Data Preprocessing**: Images were resized, normalized, and augmented to improve model generalization.
3. **Modeling**: A CNN was designed using TensorFlow to classify the images. The model was trained and evaluated for accuracy.
4. **Web App Development**: FastAPI was used to create the backend, and HTML/CSS were used for the frontend to allow users to upload an image and get a prediction.
5. **Deployment**: The application was deployed using CI/CD pipelines with GitHub Actions.

## Features
- **Image Classification**: Upload an image of a potato leaf to get a prediction of its condition.
- **Web Interface**: Simple and intuitive web interface for users to interact with the model.
- **FastAPI Backend**: Lightweight backend using FastAPI, ensuring fast responses.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mananvijay72/Potato-Disease-Classifier.git
   cd Potato-Disease-Classifier
   ```

## Usage
- **Upload Image**: Navigate to the home page and upload a potato leaf image.
- **View Prediction**: The model will display the predicted class (Early Blight, Late Blight, or Healthy) along with the confidence score.

## Model Details

- Architecture: Convolutional Neural Network (CNN)
- Framework: TensorFlow
- Input Size: 256x256 pixels
- Classes: Early Blight, Late Blight, Healthy

## Results
The model achieves high accuracy in predicting the disease class from potato leaf images. Below are the details of the model's performance:

- Accuracy: 91%
- Precision: 93%
- Recall: 99%


## Acknowledgments
Special thanks to the open-source community for providing tools and resources to make this project possible.