# Pneumonia Detection using Deep Learning

## Overview

This project aims to detect pneumonia in chest X-ray images using deep learning. Pneumonia is a dangerous respiratory disease that may affect one or both lungs and can be caused by viruses, fungi, or bacteria. The detection model is based on Convolutional Neural Network (CNN) architecture, specifically utilizing the VGG16 model with transfer learning.

## Dataset

The chest X-ray dataset used in this project is obtained from Kaggle. It consists of images categorized into two classes: "Pneumonia" and "Normal". The dataset is divided into training, testing, and validation sets.

## Tools and Technologies

- **VGG16**: A widely used CNN architecture designed for ImageNet, used here for feature extraction.
- **Transfer Learning**: Technique involving a pre-trained neural network (VGG16) and adapting it for pneumonia detection.
- **Keras and TensorFlow**: Deep learning libraries used for model development and training.
- **SciPy**: Python module for scientific computing, utilized for image transformations.

## Model Architecture

The model architecture involves taking the pre-trained VGG16 model, modifying the top layers for the pneumonia detection task, and freezing the pre-trained layers to leverage transfer learning.

## Implementation Steps

1. **Dataset Download**: Download the chest X-ray dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.

2. **Model Development**: Use the provided Python script `Pneumonia.py` for creating and training the pneumonia detection model. Adjust hyperparameters, such as epochs and batch size, based on your requirements.

3. **Model Evaluation**: Evaluate the trained model's performance using the provided Python script `Test.py`. Test the model on new chest X-ray images to check its pneumonia detection capabilities.

4. **Save and Load Model**: The trained model is saved as `our_model.h5` for future use, allowing you to load the model without retraining.

