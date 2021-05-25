import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.metrics import roc_curve
import seaborn as sns; sns.set_theme()

from LogAnalyzer.models.LSTMAutoencoder import LstmAutoencoder
from LogAnalyzer.utils.Preprocessing import preprocessData
from LogAnalyzer.utils.HelperFunctions import getDevice, getBestCut, confusion_matrix
from LogAnalyzer.utils.Plots import roc_curve_plot
import pickle

# batch size not available yet TODO: solve it
PARAMS = {
    'epochs': 20,
    'learning_rate': 0.0003,
    'loss_function': F.mse_loss,
    'opt_func': torch.optim.Adam,
    'train_anomaly_samples':0,
    'train_normal_samples': 100000,
    'val_anomaly_samples': 10000,
    'val_normal_samples': 10000,
    'decoder_hidden_dim': 150,
    'encoder_hiddem_dim:': 150,
    'encoder_out_dim': None,
    'batch_size': 1,
}
EXPERIMENT_ID = "005" + "AE"
DATA = "../data/logdata.npy"
TARGET = "../data/loglabel.npy"
GPU_CARD = 0

file = open('../experiments/' + EXPERIMENT_ID + 'PARAMS.pkl')
pickle.dump(PARAMS, file)
file.close()

X = np.load(DATA, allow_pickle=True)
Y = np.load(TARGET, allow_pickle=True)

device = getDevice(GPU_CARD)

train_data, val_data = preprocessData(
                                        X,
                                        Y,
                                        PARAMS["val_anomaly_samples"],
                                        PARAMS["val_normal_samples"],
                                        PARAMS["train_anomaly_samples"],
                                        PARAMS["train_normal_samples"],
                                        PARAMS['batch_size'],
                                        train_device=device
                                        ) 

train_data, epoch_val_data = random_split(train_data,  [90000, 10000])
train_loader = DataLoader(train_data, 1, shuffle=True)
epoch_val_loader = DataLoader(epoch_val_data, 1)

val_loader = DataLoader(val_data, 1)
val_target = val_data.getTarget()

model = LstmAutoencoder( 100, PARAMS['encoder_hidden_dim'], PARAMS['decoder_hidden_dim'])

history = model.fit(
                    PARAMS['epochs'],
                    PARAMS['learning_rate'],
                    train_loader, 
                    epoch_val_loader,
                    opt_func=PARAMS['opt_func']
                    )

#SAVING TODO: save just weights, not the models
torch.save(model, '../experiments/' + EXPERIMENT_ID + '/model.pt')
    
# Validation
model.cpu()
losses = model.predict(val_loader)

fpr, tpr, thresholds = roc_curve(val_target, losses, pos_label=1)
roc_curve_plot(fpr, tpr, savePath='../experiments/' + EXPERIMENT_ID + '/ROCCurve.png')

best_treshold = getBestCut(tpr, fpr, thresholds)
modelPred = model.predictLabels(model, val_loader, best_treshold)
cm = confusion_matrix(modelPred, val_target)
confmatrix_plot = sns.heatmap(  
                                cm,
                                robust=True,
                                annot=True,
                                center=5000,
                                linewidths= 0.2,
                                cmap="crest",
                                xticklabels=["Normal", "Anomaly"],
                                yticklabels=["Normal", "Anomaly"], 
                                fmt="g"
                                )

confmatrix_plot.savefig('../experiments/' + EXPERIMENT_ID + 'confmatrix.png')

accuracy = (cm[0,0].item() + cm[1,1].item())/ (cm[0,0].item() + cm[1,1].item() + cm[0,1].item() + cm[1,0].item())
specificity = cm[0,0].item()/( cm[1,0].item() + cm[0,0].item())
recall = cm[1,1].item()/( cm[1,1].item() + cm[0,1].item())

metrics = {
            "Accuracy":accuracy,
            "Recall": recall,
            "Specificity": specificity,
            }

file = open('../experiments/' + EXPERIMENT_ID + 'Scores.pkl')
pickle.dump(metrics, file)
file.close()

# TODO: create plots for compare loss between anomaly, and normal data.