import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRNNClassifier(nn.Module):
    def __init__(self, feature_length, hidden_length, word_length, type_num = 5, weight = None, layer = 1, batch_first = True, drop_out = 0.5):
        super(CustomRNNClassifier, self).__init__()
        self.feature_length = feature_length
        self.hidden_length = hidden_length
        self.word_length = word_length
        self.layer = layer
        self.dropout = nn.Dropout(drop_out)

        # Initialize the embedding layer, optionally with pre-trained weights
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(word_length, feature_length))
            self.embedding = nn.Embedding(num_embeddings = word_length, embedding_dim = feature_length, _weight = x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings = word_length, embedding_dim = feature_length, _weight = weight).cuda()

        # Initialize the LSTM layer
        self.lstm = nn.LSTM(input_size = feature_length,
                            hidden_size = hidden_length,
                            num_layers = layer,
                            batch_first = batch_first,
                            dropout = drop_out).cuda()

        # Initialize the fully connected layer
        self.fc = nn.Linear(hidden_length, type_num).cuda()
        self.dropout_fc = nn.Dropout(drop_out)

    def forward(self, x):
        # Convert input to LongTensor and move to CUDA device
        x = torch.LongTensor(x).cuda()

        # Apply dropout to the embedded input
        embedded = self.embedding(x)
        out_put = self.dropout(embedded)

        # Perform LSTM forward pass
        h0 = torch.randn(self.layer, out_put.size(0), self.hidden_length).cuda()
        c0 = torch.randn(self.layer, out_put.size(0), self.hidden_length).cuda()

        # LSTM前向传播
        out_put, _ = self.lstm(out_put, (h0, c0))

        # Use the output of the last time step for classification
        out_put = self.fc(out_put[:, -1, :])
        out_put = self.dropout_fc(out_put)

        return out_put


class CustomCNNClassifier(nn.Module):
    def __init__(self, feature_length, word_length, longest, kernel_length = 50, type_num = 5, weight = None, drop_out = 0.5):
        super(CustomCNNClassifier, self).__init__()
        self.feature_length = feature_length
        self.word_length = word_length
        self.kernel_length = kernel_length
        self.longest = longest
        self.dropout = nn.Dropout(drop_out)

        # Initialize the embedding layer, optionally with pre-trained weights
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(word_length, feature_length))
            self.embedding = nn.Embedding(num_embeddings = word_length, embedding_dim = feature_length, _weight = x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings = word_length, embedding_dim = feature_length, _weight = weight).cuda()

        # Initialize the convolutional layers with LeakyReLU activation function
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, longest, (2, feature_length), padding = (1, 0)),
            nn.LeakyReLU(negative_slope = 0.01)  # Using LeakyReLU with a slope of 0.01
        ).cuda()
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, longest, (3, feature_length), padding = (1, 0)),
            nn.LeakyReLU(negative_slope = 0.01)
        ).cuda()
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, longest, (4, feature_length), padding = (2, 0)),
            nn.LeakyReLU(negative_slope = 0.01)
        ).cuda()
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, longest, (5, feature_length), padding = (2, 0)),
            nn.LeakyReLU(negative_slope = 0.01)
        ).cuda()

        # Initialize the fully connected layer with dropout
        self.fc = nn.Linear(4 * longest, type_num).cuda()
        self.dropout_fc = nn.Dropout(drop_out)

    def forward(self, x):

        # Convert input to LongTensor, add a channel dimension, and move to CUDA device
        x = torch.LongTensor(x).cuda()
        out_put = self.embedding(x).view(x.shape[0], 1, x.shape[1], self.feature_length)
        out_put = self.dropout(out_put)

        # Apply convolution, activation, and pooling
        conv1 = self.conv1(out_put).squeeze(3)
        pool1 = F.max_pool1d(conv1, conv1.shape[2])

        conv2 = self.conv2(out_put).squeeze(3)
        pool2 = F.max_pool1d(conv2, conv2.shape[2])

        conv3 = self.conv3(out_put).squeeze(3)
        pool3 = F.max_pool1d(conv3, conv3.shape[2])

        conv4 = self.conv4(out_put).squeeze(3)
        pool4 = F.max_pool1d(conv4, conv4.shape[2])

        # Concatenate the pooled outputs and pass through the fully connected layer
        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2)
        out_put = self.fc(pool)
        out_put = self.dropout_fc(out_put)

        return out_put
