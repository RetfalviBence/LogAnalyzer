import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.metrics import roc_curve
import seaborn as sns; sns.set_theme()
import os

from ..models.LSTMAutoencoder import LstmAutoencoder
from ..utils.Preprocessing import preprocessData
from ..utils.HelperFunctions import getDevice, getBestCut, confusion_matrix, saveDict
from ..utils.Plots import roc_curve_plot

# batch size not available yet TODO: solve it

print("script start ...")
PARAMS = {
    'epochs': 5,
    'learning_rate': 0.001,
    'loss_function': F.mse_loss,
    'opt_func': torch.optim.Adam,
    'train_anomaly_samples':0,
    'train_normal_samples': 30000,
    'val_anomaly_samples': 5000,
    'val_normal_samples': 5000,
    'decoder_hidden_dim': 150,
    'encoder_hidden_dim': 150,
    'encoder_out_dim': None,
    'batch_size': 32,
}
EXPERIMENT_ID = "006" + "AE"
DATA = "LogAnalyzer/data/logdata.npy"
TARGET = "LogAnalyzer/data/loglabel.npy"
GPU_CARD = 0
experimentPath = "LogAnalyzer/experiments/" + EXPERIMENT_ID

if not os.path.isdir("LogAnalyzer/experiments"):
  os.mkdir("LogAnalyzer/experiments")

if not os.path.isdir(experimentPath):
  os.mkdir(experimentPath)

saveDict(PARAMS, experimentPath + '/PARAMS.pkl')

X = np.load(DATA, allow_pickle=True)
Y = np.load(TARGET, allow_pickle=True)
print("data loaded")

device = getDevice(GPU_CARD)
print("device selected : ", device)

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

train_data, epoch_val_data = random_split(train_data,  [25000, 5000])
train_loader = DataLoader(train_data, PARAMS['batch_size'], shuffle=True)
epoch_val_loader = DataLoader(epoch_val_data, 1)
print("data loaded successfully")

val_loader = DataLoader(val_data, 1)
val_target = val_data.getTarget()
print("data preprocessed.")

model = LstmAutoencoder( 100, PARAMS['encoder_hidden_dim'], PARAMS['decoder_hidden_dim'])
model.to(device)

print("start training")
history = model.fit(
                    PARAMS['epochs'],
                    PARAMS['learning_rate'],
                    train_loader, 
                    epoch_val_loader,
                    opt_func=PARAMS['opt_func']
                    )

print("finish training")
#SAVING TODO: save just weights, not the models
np.save(experimentPath + '/learningHistory', history)
torch.save(model, experimentPath + '/model.pt')
    
# Validation
model.cpu()
losses = model.predict(val_loader)

fpr, tpr, thresholds = roc_curve(val_target, losses, pos_label=1)
roc_curve_plot(fpr, tpr, savePath=experimentPath + '/ROCCurve.png')

best_treshold = getBestCut(tpr, fpr, thresholds)
modelPred = model.predictLabels(val_loader, best_treshold)
cm = confusion_matrix(modelPred, val_target)
confmatrix_plot = sns.heatmap(  
                                cm,
                                robust= True,
                                annot= True,
                                center= 5000,
                                linewidths = 0.2,
                                cmap="crest",
                                xticklabels =["Normal", "Anomaly"],
                                yticklabels =["Normal", "Anomaly"], 
                                fmt = "g"
                                )

confmatrix_plot.get_figure().savefig(experimentPath + '/confmatrix.png')

accuracy = (cm[0,0].item() + cm[1,1].item())/ (cm[0,0].item() + cm[1,1].item() + cm[0,1].item() + cm[1,0].item())
specificity = cm[0,0].item()/( cm[1,0].item() + cm[0,0].item())
recall = cm[1,1].item()/( cm[1,1].item() + cm[0,1].item())

metrics = {
            "Accuracy":accuracy,
            "Recall": recall,
            "Specificity": specificity,
            }

saveDict(metrics, experimentPath + '/Scores.pkl')

# TODO: create plots for compare loss between anomaly, and normal data.