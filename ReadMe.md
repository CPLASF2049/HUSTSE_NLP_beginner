# Text Classification Project

## Introduction
This project utilizes deep learning techniques to classify texts into predefined categories. It's designed to demonstrate the power of recurrent neural networks (RNN) and convolutional neural networks (CNN) in the context of natural language processing.

## Functionality
The codebase implements the following functionalities:
- Text preprocessing and augmentation to enhance model performance.
- Training of RNN and CNN models using either random or GloVe word embeddings.
- Visualization of training and testing loss and accuracy over iterations.

## Usage
### System Requirements
- Python 3.x
- PyTorch
- Matplotlib
- NumPy
- CUDA (for GPU acceleration, if available)

### Deployment
1. Ensure that Python and the required packages are installed.
2. Place the project files in a directory of your choice.
3. Adjust the file paths in the `main.py` script to match your local setup.

### Operation Instructions
- Run `main.py` to execute the program. This will read the training data, prepare the embeddings, train the models, and generate visualizations.

Before running the program, ensure that you have downloaded the necessary datasets and word embeddings. The training dataset `train.tsv` and a compressed file containing pre-trained GloVe word embeddings (`glove.6B.zip`) are not included in the repository due to size constraints. You can download them from the following links:

- [Download train.tsv](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) 
- [Download glove.6B.zip](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip) 

After downloading, extract the `glove.6B.zip` file to access `glove.6B.50d.txt` and other included files. Please make sure to place these files in the appropriate directory as specified in the `main.py` script.

## Code Organization
The project is organized into the following files and directories:
- `main.py`: The entry point of the project, responsible for reading data, initializing models, and running the training and evaluation process.
- `model_training_and_visualization.py`: Contains the functions for training the models and visualizing the results.
- `data_preprocessing_and_augmentation.py`: Includes classes and functions for data preprocessing, augmentation, and batch creation.
- `network_models.py`: Defines the custom neural network architectures used in the project.
- `train.tsv`: The training dataset in TSV format.
- `glove.6B.50d.txt`: Pre-trained GloVe word embeddings.