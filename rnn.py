import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 2
        self.rnn = nn.LSTM(input_dim, h, self.numOfLayer, batch_first=False)
        self.W = nn.Linear(h, 5)  # 5 output classes for star ratings (0-4)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.dropout = nn.Dropout(0.3)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        '''
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        _, hidden = self.rnn(inputs)
        
        # [to fill] obtain output layer representations
        hidden = self.dropout(hidden[-1])
        output = self.W(hidden)
        # [to fill] sum over output 

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)
        '''
        _, (hidden_state, _) = self.rnn(inputs)
    
        # Get the final hidden state from the last layer
        final_hidden = hidden_state[-1]  # shape: [batch_size, hidden_size]
        
        # Apply dropout
        final_hidden = self.dropout(final_hidden)
        
        # Apply linear layer
        output = self.W(final_hidden)
        
        # Apply softmax
        predicted_vector = self.softmax(output)
    
        return predicted_vector


def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    val = []
    tes = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        tes.append((elt["text"].split(),int(elt["stars"]-1)))
        
    return tra, val, tes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", required = True, help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            
            # Collect all examples in a minibatch
            batch_inputs = []
            batch_lengths = []
            batch_labels = []
            
            # Process each example in the minibatch
            for example_index in range(minibatch_size):
                idx = minibatch_index * minibatch_size + example_index
                input_words, gold_label = train_data[idx]
                input_words = " ".join(input_words)
                
                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                
                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                
                # Keep track of original sequence length
                batch_lengths.append(len(vectors))
                
                # Add to batch inputs
                batch_inputs.append(np.array(vectors))
                batch_labels.append(gold_label)
            
            # Find the max sequence length in this batch
            max_length = max(batch_lengths)
            
            # Pad sequences to max_length
            padded_inputs = []
            for i, seq in enumerate(batch_inputs):
                if len(seq) < max_length:
                    # Pad with zeros
                    padding = np.zeros((max_length - len(seq), word_embedding['unk'].shape[0]))
                    padded_seq = np.vstack((seq, padding))
                else:
                    padded_seq = seq
                padded_inputs.append(padded_seq)
            
            # Stack all sequences into a single array - shape: [max_length, batch_size, features]
            batch_tensor = np.stack([p for p in padded_inputs], axis=1)
            
            # Convert to tensor
            batch_tensor = torch.tensor(batch_tensor, dtype=torch.float32)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            
            # Forward pass
            output = model(batch_tensor)
            
            # Compute loss
            loss = model.compute_Loss(output, batch_labels)
            
            # Backpropagation
            loss.backward()
            
            # Optional: gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            loss_total += loss.item()
            loss_count += 1
        
        # Print epoch statistics    
        print(f"Average loss: {loss_total/loss_count:.4f}")
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        training_accuracy = correct/total

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = np.array(vectors)  # Convert list of arrays to a single numpy array
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total

        '''
        if validation_accuracy < last_validation_accuracy and training_accuracy > last_train_accuracy or epoch + 1 >= args.epochs:
            stopping_condition = True
            if epoch + 1 >= args.epochs:
                print(f"Reached maximum number of epochs ({args.epochs})")
            else:
                print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy
        '''
        
        epoch += 1
        
        if epoch == args.epochs:
            stopping_condition = True
        
            

    print("========== Testing model on test data ==========")
    model.eval()
    # Reset counters here
    correct = 0 
    total = 0
    print("Testing started...")
    for input_words, gold_label in tqdm(test_data):
        input_words = " ".join(input_words)
        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                   in input_words]

        vectors = np.array(vectors)
        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)
        predicted_label = torch.argmax(output)
        correct += int(predicted_label == gold_label)
        total += 1
    
    test_accuracy = correct / total
    print("Testing completed")
    print("Test accuracy: {}".format(test_accuracy))


    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance