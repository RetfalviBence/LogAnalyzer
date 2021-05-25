import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.metrics import roc_curve
import seaborn as sns; sns.set_theme()

from LogAnalyzer.models.LogRobust import LOG_ROBUST
from LogAnalyzer.utils.Preprocessing import preprocessData
from LogAnalyzer.utils.HelperFunctions import getDevice, getBestCut, confusion_matrix
from LogAnalyzer.utils.Plots import roc_curve_plot
import pickle

# batch size not available yet TODO: solve it
PARAMS = {
    'epochs': 20,
    'learning_rate': 0.0001,
    'opt_func': torch.optim.Adam,
    'train_anomaly_samples':10000,
    'train_normal_samples': 10000,
    'val_anomaly_samples': 3000,
    'val_normal_samples': 3000,
    'hidden_size': 64,
    'batch_size': 256,
}
EXPERIMENT_ID = "005" + "LR"
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

train_loader = DataLoader(train_data, 1, shuffle=True)
val_loader = DataLoader(val_data, 1)

model = LOG_ROBUST( 100, PARAMS['hidden_size'], PARAMS['batch_size'])
model.to(device)

history = model.fit(
                    PARAMS['epochs'],
                    PARAMS['learning_rate'],
                    train_loader, 
                    val_loader,
                    opt_func=PARAMS['opt_func']
                    )

#SAVING TODO: save just weights, not the models
torch.save(model, '../experiments/' + EXPERIMENT_ID + '/model.pt')
    
# Validation
model.cpu()
probas, labels = model.pred_probas(val_data)

fpr, tpr, thresholds = roc_curve(labels, probas[:,1], pos_label=1)
roc_curve_plot(fpr, tpr, savePath='../experiments/' + EXPERIMENT_ID + '/ROCCurve.png')

best_treshold = getBestCut(tpr, fpr, thresholds)
modelPred, labels = model.pred(val_data, best_treshold)

cm = confusion_matrix(modelPred, labels)
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