# Butterfly Classification with CNN

This repository contains a convolutional neural network (CNN) model to solve a classification problem using images of butterflies. The goal is to predict the type of butterfly in the images. This project is part of a competition hosted on Kaggle.

## Important note:
You cannot submit the code to the Kaggle competition without first modifying the prediction output.

## Dataset

The dataset used for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/code).

The dataset consists of:
- **Training set**: Images and labels for training the model.
- **Testing set**: Images for evaluating the model.
- **Data/**: Contains the training and testing images along with their respective CSV files.
- **main.ipynb**: Jupyter notebook containing the entire workflow from data preprocessing to model evaluation.

## Required Libraries

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.
- `tensorflow` and `keras`: For building and training the CNN model.
- `scikit-learn`: For data splitting.
- `Pillow`: For image processing.

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/code).

2. Open the Jupyter notebook `main.ipynb` and run all the cells to execute the following steps:
   - Load and preprocess the data.
   - Visualize the data distribution.
   - Build and compile the CNN model.
   - Train the model.
   - Evaluate the model.
   - Make predictions on the test set.

## Model Architecture

The CNN model is built using TensorFlow and Keras. The architecture consists of:

- Convolutional layers with ReLU activation and MaxPooling.
- Dense layers with ReLU activation.
- Dropout layer for regularization.
- Softmax activation for the output layer to classify the butterfly species.


## Results

- The model is trained for 25 epochs with data augmentation.
- The training and validation accuracy and loss are plotted to visualize the model's performance.
