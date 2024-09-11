import csv
import random

# Import custom classes for data preprocessing and augmentation
from data_preprocessing_and_augmentation import RandomEmbeddingFeature, GloveEmbeddingFeature

# Import the function for evaluating and visualizing model results
from model_training_and_visualization import evaluate_result_chart

# Read training data
with open('train.tsv') as f:
    tsvreader = csv.reader (f, delimiter = '\t')
    temp = list ( tsvreader )

# Read pre-trained GloVe word embeddings
with open('glove.6B.50d.txt','rb') as f:
    lines = f.readlines()

# Create a dictionary to hold the GloVe word embeddings
trained_dict = dict()
n = len(lines)
for i in range(n):
    line=lines[i].split()
    trained_dict[line[0].decode("utf-8").upper()] = [float(line[j]) for j in range(1,51)]

# Set hyperparameters for the training process
iteration_times = 50
learning_rate = 0.0015
dropout_rate = 0.5

data = temp[1:]
batch_size = 500

# Create an instance of RandomEmbeddingFeature to handle random embeddings
random.seed(2024)
random_embedding = RandomEmbeddingFeature(data = data)
random_embedding.get_words()
random_embedding.get_id()

# Create an instance of GloveEmbeddingFeature to handle GloVe embeddings
random.seed(2024)
glove_embedding = GloveEmbeddingFeature(data = data, trained_dict = trained_dict)
glove_embedding.get_words()
glove_embedding.get_id()

# Call the function to evaluate and visualize the results of different embeddings and models
evaluate_result_chart(random_embedding, glove_embedding, learning_rate, batch_size, iteration_times, dropout_rate)