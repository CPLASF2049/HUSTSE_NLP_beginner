import matplotlib.pyplot
import torch
import torch.nn.functional as F
from torch import optim
from network_models import CustomRNNClassifier, CustomCNNClassifier
from data_preprocessing_and_augmentation import get_batch

# Function to train the model and record the training and testing loss and accuracy
def embed(model, train,test, learning_rate, iter_times):
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_fun = F.cross_entropy
    train_loss_record = list()
    test_loss_record = list()
    long_loss_record = list()
    train_record = list()
    test_record = list()
    long_record = list()

    for iteration in range(iter_times): # Loop over the number of iterations
        model.train()
        for i, batch in enumerate(train): # Loop over the training batches
            x, y = batch
            y = y.cuda()
            pred = model(x).cuda()
            optimizer.zero_grad()
            loss = loss_fun(pred, y).cuda()
            loss.backward()
            optimizer.step()

        model.eval()
        # Initialize lists to store loss and accuracy for training and testing sets
        train_accuracy = list()
        test_accuracy = list()
        long_accuracy = list()
        train_loss = 0
        test_loss = 0
        long_loss = 0

        # Calculate loss and accuracy for the training set
        for i, batch in enumerate(train):
            x, y = batch
            y = y.cuda()
            pred = model(x).cuda()
            loss = loss_fun(pred, y).cuda()
            train_loss += loss.item()
            _, y_pre = torch.max(pred, -1)
            accuracy = torch.mean((y_pre == y).float().clone().detach())
            train_accuracy.append(accuracy)

        # Calculate loss and accuracy for the testing set
        for i, batch in enumerate(test):
            x, y = batch
            y = y.cuda()
            pred = model(x).cuda()
            loss = loss_fun(pred, y).cuda()
            test_loss += loss.item()
            _, y_pre = torch.max(pred, -1)
            accuracy = torch.mean((y_pre == y).float().clone().detach())
            test_accuracy.append(accuracy)
            if(len(x[0])) > 20:
                long_accuracy.append(accuracy)
                long_loss += loss.item()

        # Calculate average loss and accuracy and append to the records
        trains_accuracy = sum(train_accuracy) / len(train_accuracy)
        tests_accuracy = sum(test_accuracy) / len(test_accuracy)
        longs_accuracy = sum(long_accuracy) / len(long_accuracy)

        train_loss_record.append(train_loss / len(train_accuracy))
        test_loss_record.append(test_loss / len(test_accuracy))
        long_loss_record.append(long_loss / len(long_accuracy))
        train_record.append(trains_accuracy.cpu())
        test_record.append(tests_accuracy.cpu())
        long_record.append(longs_accuracy.cpu())

        print("---------- Iteration", iteration + 1, "----------")
        print("Train Loss:", train_loss/ len(train_accuracy))
        print("Test Loss:", test_loss/ len(test_accuracy))
        print("Train Accuracy:", trains_accuracy)
        print("Test Accuracy:", tests_accuracy)
        print("Long Sentence Accuracy:", longs_accuracy)

    return train_loss_record, test_loss_record, long_loss_record, train_record, test_record, long_record

# Function to evaluate and plot the results of the models with random and glove embeddings
def evaluate_result_chart(random_embedding, glove_embedding, learning_rate, batch_size, iter_times, dropout_rate):

    # Get batches for training and testing for both random and glove embeddings
    train_random = get_batch(random_embedding.train_matrix, random_embedding.train_y, batch_size)
    test_random = get_batch(random_embedding.test_matrix, random_embedding.test_y, batch_size)
    train_glove = get_batch(glove_embedding.train_matrix, glove_embedding.train_y, batch_size)
    test_glove = get_batch(random_embedding.test_matrix, glove_embedding.test_y, batch_size)

    # Initialize seeds for reproducibility
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)

    # Initialize models with random embeddings
    random_rnn = CustomRNNClassifier(50, 50, random_embedding.len_words, drop_out = dropout_rate)
    random_cnn = CustomCNNClassifier(50, random_embedding.len_words, random_embedding.longest, drop_out = dropout_rate)

    # Initialize models with glove embeddings
    glove_rnn = CustomRNNClassifier(50, 50, glove_embedding.len_words, weight = torch.tensor(glove_embedding.embedding, dtype = torch.float), drop_out = dropout_rate)
    glove_cnn = CustomCNNClassifier(50, glove_embedding.len_words, glove_embedding.longest,weight = torch.tensor(glove_embedding.embedding, dtype = torch.float), drop_out = dropout_rate)

    # Get results for random RNN and CNN models
    trl_ran_rnn,tel_ran_rnn,lol_ran_rnn,tra_ran_rnn,tes_ran_rnn,lon_ran_rnn = embed(random_rnn,train_random,test_random,learning_rate,  iter_times)
    trl_ran_cnn,tel_ran_cnn,lol_ran_cnn, tra_ran_cnn, tes_ran_cnn, lon_ran_cnn = embed(random_cnn, train_random,test_random, learning_rate, iter_times)

    # Get results for glove RNN and CNN models
    trl_glo_rnn,tel_glo_rnn,lol_glo_rnn, tra_glo_rnn, tes_glo_rnn, lon_glo_rnn = embed(glove_rnn, train_glove,test_glove, learning_rate, iter_times)
    trl_glo_cnn,tel_glo_cnn,lol_glo_cnn, tra_glo_cnn, tes_glo_cnn, lon_glo_cnn = embed(glove_cnn,train_glove,test_glove, learning_rate, iter_times)

    # Plotting the results in a 2x2 grid fashion
    x = list(range(1,iter_times+1))
    matplotlib.pyplot.subplot(2, 2, 1)
    matplotlib.pyplot.plot(x, trl_ran_rnn, 'r--', label='RNN+Random')
    matplotlib.pyplot.plot(x, trl_ran_cnn, 'g--', label='CNN+Random')
    matplotlib.pyplot.plot(x, trl_glo_rnn, 'b--', label='RNN+Glove')
    matplotlib.pyplot.plot(x, trl_glo_cnn, 'y--', label='CNN+Glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 2)
    matplotlib.pyplot.plot(x, tel_ran_rnn, 'r--', label='RNN+Random')
    matplotlib.pyplot.plot(x, tel_ran_cnn, 'g--', label='CNN+Random')
    matplotlib.pyplot.plot(x, tel_glo_rnn, 'b--', label='RNN+Glove')
    matplotlib.pyplot.plot(x, tel_glo_cnn, 'y--', label='CNN+Glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 3)
    matplotlib.pyplot.plot(x, tra_ran_rnn, 'r--', label='RNN+Random')
    matplotlib.pyplot.plot(x, tra_ran_cnn, 'g--', label='CNN+Random')
    matplotlib.pyplot.plot(x, tra_glo_rnn, 'b--', label='RNN+Glove')
    matplotlib.pyplot.plot(x, tra_glo_cnn, 'y--', label='CNN+Glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 2, 4)
    matplotlib.pyplot.plot(x, tes_ran_rnn, 'r--', label='RNN+Random')
    matplotlib.pyplot.plot(x, tes_ran_cnn, 'g--', label='CNN+Random')
    matplotlib.pyplot.plot(x, tes_glo_rnn, 'b--', label='RNN+Glove')
    matplotlib.pyplot.plot(x, tes_glo_cnn, 'y--', label='CNN+Glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8, forward = True)
    matplotlib.pyplot.savefig('main_plot.jpg')
    matplotlib.pyplot.show()
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.plot(x, lon_ran_rnn, 'r--', label='RNN+Random')
    matplotlib.pyplot.plot(x, lon_ran_cnn, 'g--', label='CNN+Random')
    matplotlib.pyplot.plot(x, lon_glo_rnn, 'b--', label='RNN+Glove')
    matplotlib.pyplot.plot(x, lon_glo_cnn, 'y--', label='CNN+Glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Long Sentence Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(x, lol_ran_rnn, 'r--', label='RNN+Random')
    matplotlib.pyplot.plot(x, lol_ran_cnn, 'g--', label='CNN+Random')
    matplotlib.pyplot.plot(x, lol_glo_rnn, 'b--', label='RNN+Glove')
    matplotlib.pyplot.plot(x, lol_glo_cnn, 'y--', label='CNN+Glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Long Sentence Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")

    # Save and show the plots
    matplotlib.pyplot.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8, forward = True)
    matplotlib.pyplot.savefig('sub_plot.jpg')
    matplotlib.pyplot.show()