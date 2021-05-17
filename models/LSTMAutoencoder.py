#import time
import numpy as np
import pandas as pd
import torch
import copy
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


# functionality, 1 linear layer, zero linear layer

class Encoder(nn.Module):
    """ Encoder part of the autoencoder """
    def __init__(self, input_dim, lstm_hidden_dim, encoder_out_dim=None, batch_size=1, num_layers=1):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.encoder_out_dim = encoder_out_dim # dimension of the embedding created by the encoder, if use extra linear layer
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # Define Linear layer
        if self.encoder_out_dim is not None:
            self.enocode_linear = nn.Linear(self.hidden_dim, self.encoder_out_dim, bias=True)
        
        # Define LSTM layer
        self.encode_lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        
    def forward(self, inputs):
        """ Create the encoded vector for a data instance """
        lstm_out, (embedding, _) = self.encode_lstm(inputs)
        
        if self.encoder_out_dim is not None:
            embedding = self.enocode_linear(embedding)
        
        return embedding
    
class Decoder(nn.Module):
    """ Decoder part of the autoencoder """
    def __init__(self, input_dim, lstm_hidden_dim, reconstruction_dim, batch_size, num_layers=1):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.reconstruction_dim = reconstruction_dim # dimension of the vectors, that we would reconstruct
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # Define LSTM layer
        self.encode_lstm = nn.LSTM(self.input_dim, self.lstm_hidden_dim, self.num_layers, batch_first=True)
        
        # Linear layer to create predictions
        self.predictor = nn.Linear(self.decoder_hidden_dim, self.reconstruction_dim, bias=True)
        
    def forward(self, inputs, seq_len):
        """ Create rekonsturction vectors """
        # create input for all lstm time step
        repeated_embedding = inputs.repeat(1, seq_len, 1)
        lstm_out, (embedding, _) = self.encode_lstm(repeated_embedding)
        
        # flip the direction of the predicts.
        out = lstm_out.flip((1))
        out = self.predictor(out)
        return out
    
class LstmAutoencoder(nn.Module):
    """ Lstm autoencoder, combination of the encoder, decoder """
    def __init__(self, input_dim, encoder_hidden_dim, decoder_hidden_dim, encoder_out_dim=None, batch_size=1, num_layers=1):
        super(LstmAutoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, encoder_hidden_dim, encoder_out_dim, batch_size, num_layers)
        self.decoder = Decoder(encoder_out_dim, decoder_hidden_dim, input_dim, batch_size)
        
    def forward(self, inputs):
        seq_len = inputs.shape[1]
        embedding = self.encoder(inputs)
        decoded_input = self.decoder(embedding, seq_len)
        return decoded_input
        
    def step(self, batch, lossFunction):
        inputs, labels = batch
        out = self(inputs)
        loss = lossFunction(out, inputs)
        return loss
        
    def evaluate(self, val_loader):
        """Evaluate the model's performance on the validation set"""
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
        
    def fit(self, epochs, lr, train_loader, val_loader, opt_func=torch.optim.SGD, lossFunction=F.mse_loss):
        """Train the model using gradient descent"""
        history = []
        optimizer = opt_func(self.parameters(), lr)
        for epoch in range(epochs):
            for batch in train_loader:
                loss = self.step(batch, lossFunction)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
            val_history = [self.step(batch, lossFunction) for batch in val_loader]
            val_loss = torch.stack(val_history).mean()
            print("Epoch [{}], val_loss: {:.4f}".format(epoch, val_loss))
            history.append(val_loss)
    return history
    
    def predict(self, dataset, criterion=nn.L1Loss):
        """ Return with predictions for input vectors, and losses of predictions """
        predictions, losses = [], []
        with torch.no_grad():
            model = self.eval()
            for seq_true in dataset:
                seq_true = torch.reshape(seq_true, (1,-1,100))
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                predictions.append(seq_pred.numpy().flatten())
                losses.append(loss.item())
        return predictions, losses
        
    def predictLabels(self, dataset, treshold, criterion=nn.L1Loss):
        """ return with the predicted label, for all data inistances """
        predictions, losses = self.predict(dataset, criterion)
        model_prediction = np.array([x > treshold for x in losses])
        return model_prediction.astype(int)