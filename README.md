# Multi-Layer Perceptron (MLP) Classifier on Fashion MNIST Dataset

## Overview

This repository contains Python scripts that demonstrate the use of MLPClassifier from scikit-learn to classify images from the Fashion MNIST dataset. Various configurations of the MLP model are explored by adjusting parameters such as hidden layer sizes, number of iterations, and learning rate.

## Dataset

The Fashion MNIST dataset consists of 60,000 training images and 10,000 test images across 10 classes. Each image is grayscale with dimensions 28x28 pixels.

## Code Explanation

1. **Loading and Preprocessing Data**
   - The Fashion MNIST dataset is loaded using `fashion_mnist.load_data()` from TensorFlow.
   - Data shapes for training and test sets are printed.

2. **Data Visualization**
   - A subset of images (36 in total) from the training set is visualized using matplotlib.

3. **Model Training and Evaluation**
   - **Initial Configuration:**
     - MLPClassifier is initialized with a single hidden layer of size 4.
     - The model is trained and evaluated using default parameters.
   
   - **Parameter Tuning:**
     - Several configurations are tested by adjusting parameters:
       - Increasing hidden layer sizes to (64, 64).
       - Increasing the number of iterations to 500.
       - Adjusting the learning rate (`alpha`) to 0.001.
       - Combining all three adjustments: hidden layers (64, 64), iterations (500), and learning rate (0.001).

4. **Model Evaluation Metrics**
   - **Confusion Matrix:**
     - Confusion matrices are computed and printed for each model configuration to visualize predictions versus actual labels.

   - **Accuracy Calculation:**
     - The accuracy of each model configuration is calculated using a custom function `accuracy()`.
   
   - **Classification Report:**
     - Classification reports are generated to provide precision, recall, F1-score, and support for each class.

## Usage

Ensure Python environment is set up with necessary libraries (`numpy`, `pandas`, `matplotlib`, `tensorflow`, `scikit-learn`). Run the script to load the dataset, train the MLP models, and evaluate their performance on Fashion MNIST.

## Improvements

- Explore more complex architectures and hyperparameters for better performance.
- Implement cross-validation for robust model evaluation.
- Visualize feature maps or activations to gain insights into model decisions.

