import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

# Function to split data into training and testing sets
def split(data, test_rate = 0.3):
    train = list()
    test = list()
    for datum in data:
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)
    return train, test

# Class to handle random embedding features
class RandomEmbeddingFeature():
    def __init__(self, data, test_rate=0.3):
        self.dict_words = dict()
        # Sort data by the length of the sentence for length-based splitting
        data.sort(key = lambda x: len(x[2].split()))
        self.data = data
        self.len_words = 0
        self.train, self.test = split(data, test_rate=test_rate)
        self.train_y = [int(term[3]) for term in self.train]
        self.test_y = [int(term[3]) for term in self.test]
        self.train_matrix = list()
        self.test_matrix = list()
        self.longest = 0
        self.augmentation_rate = 0.1 # Rate of data augmentation
        self.augment_data()

    def get_words(self):
        # Create a word dictionary with unique word indices
        for term in self.data:
            s = term[2]
            s = s.upper()
            words = s.split()
            for word in words:  # Process every word
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)+1
        self.len_words = len(self.dict_words)

    def get_id(self):
        # Convert words to their corresponding indices based on the word dictionary
        for term in self.train:
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item)) # Update longest sentence length
            self.train_matrix.append(item)
        for term in self.test:
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_matrix.append(item)
        self.len_words += 1 # Increment word length for padding if necessary

    def augment_data(self):
        # Augment the data by inserting random words into sentences
        self.data = [
            (term[0], term[1], self.augment_sentence(term[2]), term[3])
            for term in self.data
        ]

    def augment_sentence(self, sentence):
        # Function to augment a sentence by inserting a random word
        words = sentence.split()
        if len(words) > 1 and random.random() < self.augmentation_rate:
            insert_position = random.randint(1, len(words) - 1)
            words_to_insert = random.sample(words, 1)  # Select a random word to insert
            words = words[:insert_position] + words_to_insert + words[insert_position:]
        return ' '.join(words)

# Class to handle GloVe embedding features
class GloveEmbeddingFeature():
    def __init__(self, data,trained_dict,test_rate = 0.3):
        self.dict_words = dict()
        self.trained_dict = trained_dict
        data.sort(key = lambda x:len(x[2].split()))
        self.data = data
        self.len_words = 0
        self.train, self.test = split(data, test_rate=test_rate)
        self.train_y = [int(term[3]) for term in self.train]
        self.test_y = [int(term[3]) for term in self.test]
        self.train_matrix = list()
        self.test_matrix = list()
        self.longest=0
        self.embedding=list()

    def get_words(self):
        # Create a word dictionary and populate the embedding list with GloVe vectors
        self.embedding.append([0] * 50) # Add a zero vector for padding
        for term in self.data:
            s = term[2]
            s = s.upper()
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1
                    if word in self.trained_dict:
                        self.embedding.append(self.trained_dict[word])
                    else:
                        self.embedding.append([0] * 50) # Add a zero vector for out-of-vocabulary words
        self.len_words = len(self.dict_words)

    def get_id(self):
        # Convert words to their corresponding indices and prepare data matrices
        for term in self.train:
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest=max(self.longest,len(item))
            self.train_matrix.append(item)
        for term in self.test:
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_matrix.append(item)
        self.len_words += 1

# Custom Dataset class to be used with PyTorch's DataLoader
class ClsDataset(Dataset):
    def __init__(self, sentence, emotion):
        self.sentence = sentence
        self.emotion= emotion

    def __getitem__(self, item):
        # Return a single sample from the dataset
        return self.sentence[item], self.emotion[item]

    def __len__(self):
        # Return the length of the dataset
        return len(self.emotion)

# Collate function to pad sequences and create mini-batch tensors
def collate_fn(batch_data):
    sentence, emotion = zip(*batch_data)
    sentences = [torch.LongTensor(sent) for sent in sentence]
    padded_sents = pad_sequence(sentences, batch_first = True, padding_value = 0)
    return torch.LongTensor(padded_sents), torch.LongTensor(emotion)

# Function to get data batches for training and testing
def get_batch(x,y,batch_size):
    dataset = ClsDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = False, drop_last = True, collate_fn = collate_fn)
    return dataloader
