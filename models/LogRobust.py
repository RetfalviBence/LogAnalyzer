import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn import preprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import shuffle

class LOG_ROBUST(nn.Module):
    # Log Robust implementation

    def __init__(self, input_dim, hidden_dim, batch_size, device=torch.device("cpu"), loss_weights=torch.tensor([0.5, 0.5]), num_layers=1, with_attention=True):
        super(LOG_ROBUST, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.loss_weights = loss_weights.to(device)
        self.with_attention= with_attention

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # Define a fc layer with tanh activation
        self.linear = nn.Linear(self.hidden_dim * 2, 1, bias=False)
        self.attention_activation = nn.Tanh()

        # Define an fc layer to reduct dimension, and use softmax to predict
        self.softmaxFcn = nn.Linear(self.hidden_dim*2, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, lengths=[]):
        # input can't be one data record, couse of the padded sequence
        # shape of lstm_out: [batch_size, input_length ,hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (batch_size, num_layers, hidden_dim).

        # create PackedSequence object for handle different length sequences
        padded_input = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        
        # run lstm
        lstm_out, self.hidden = self.lstm(padded_input)

        # unpack out
        unpacked_out, lengths_out = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # fcn with tanh
        if self.with_attention:
            fcn_out = self.linear(unpacked_out)
            fcn_activated = self.attention_activation(fcn_out)
            # multiply data with the created weight
            summed_state = torch.sum(fcn_activated * unpacked_out,1)
        else:
            summed_state = torch.sum(unpacked_out,1)

        # fcn with softmax
        fcn_softmax = self.softmaxFcn(summed_state)
        y_pred = self.softmax(fcn_softmax)
        
        return y_pred 

    def training_step(self, batch):
        logEmbeddings, lengths, labels = batch
        out = self(logEmbeddings, lengths)
        loss = F.cross_entropy(out, labels, self.loss_weights)

        return loss

    def validation_step(self, batch):
        logEmbeddings, lengths, labels = batch
        out = self(logEmbeddings, lengths)
        loss = F.cross_entropy(out, labels, self.loss_weights)
        acc = accuracy(out, labels)
        return {'val_loss':loss, 'val_acc':acc}
    
    def prediction_step(self, batch):
        logEmbeddings, lengths, labels = batch
        out = self(logEmbeddings, lengths)
        
        return out, labels

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
    def fit(self, epochs, lr, train_loader, val_loader, opt_func=torch.optim.Adam):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(self.parameters(), lr)
    for epoch in range(epochs):
    
        # Training Phase 
        for batch in train_loader:
            loss = self.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validation phase
        result = evaluate(self, val_loader)
        self.epoch_end(epoch, result)
        history.append(result)
        
    return history
    
    def evaluate(self, val_loader):
        """Evaluate the model's performance on the validation set"""
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def pred(self, dataSet, th=0.5):
        # TODO: create a more robust solution
        # return with the prediction value 1 or 0
        logEmbeddings = dataSet.dataset.data[dataSet.indices]
        lengths = dataSet.dataset.lengths[dataSet.indices]
        labels = dataSet.dataset.target[dataSet.indices]
   
        out = self.forward(logEmbeddings, lengths)
        values = out[:, 1]
        preds = []
    
        for value in values:
            if value > th:
            preds.append(1)
        else:
            preds.append(0)
        
        return torch.tensor(preds), labels
    
    def pred_probas(self, dataSet):
        # return with the probalities of the 2 label, anomaly or not
        logEmbeddings = dataSet.dataset.data[dataSet.indices]
        lengths = dataSet.dataset.lengths[dataSet.indices]
        labels = dataSet.dataset.target[dataSet.indices]

        out = self.forward(logEmbeddings, lengths)

        return out.detach().numpy()

# helper functions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))