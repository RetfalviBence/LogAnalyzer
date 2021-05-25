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

file = open('../experiments/' + PARAMS['experiment'] + 'PARAMS.pkl')
pickle.dump(PARAMS, file)
file.close()

EXPERIMENT_ID = "005AE"
DATA = "../data/logdata.npy"
TARGET = "../data/loglabel.npy"

def runAeExperimentAe(params=PARAMS, experiment_id=EXPERIMENT_ID, data=DATA, target=TARGET, gpu_card=0):
    X = np.load(data, allow_pickle=True)
    Y = np.load(target, allow_pickle=True)

    device = getDevice(gpu_card)

    train_data, val_data = preprocessData(
                                                X,
                                                Y,
                                                params.val_anomaly_samples,
                                                params.val_normal_samples,
                                                params.train_anomaly_samples,
                                                params.train_normal_samples,
                                                params['batch_size'],
                                                train_device=device
                                                ) 

    train_data, epoch_val_data = random_split(train_data,  [90000, 10000])
    train_loader = DataLoader(train_data, 1, shuffle=True)
    epoch_val_loader = DataLoader(epoch_val_data, 1)

    val_loader = DataLoader(val_data, 1)
    val_target = val_data.getTarget()

    model = LstmAutoencoder( 100, PARAMS['encoder_hidden_dim'], PARAMS['decoder_hidden_dim'])

    history = model.fit(
                        params['epochs'],
                        params['learning_rate'],
                        train_loader, 
                        epoch_val_loader,
                        opt_func=params['opt_func']
                        )

    #SAVING TODO: save just weights, not the models
    torch.save(model, '../experiments/' + PARAMS['experiment'] + '/model.pt')
    
    # Validation
    model.cpu()
    losses = model.predict(val_loader)

    fpr, tpr, thresholds = roc_curve(val_target, losses, pos_label=1)
    roc_curve_plot(fpr, tpr, savePath='../experiments/' + PARAMS['experiment'] + '/ROCCurve.png')

    best_treshold = getBestCut(tpr, fpr, thresholds)
    modelPred = model.predictLabels(model, val_loader, best_treshold)
    cm = confusion_matrix(modelPred, val_target)
    confmatrix_plot = sns.heatmap(  cm,
                                    robust=True,
                                    annot=True,
                                    center=5000,
                                    linewidths= 0.2,
                                    cmap="crest",
                                    xticklabels=["Normal", "Anomaly"],
                                    yticklabels=["Normal", "Anomaly"], 
                                    fmt="g"
                                    )

    confmatrix_plot.savefig('../experiments/' + PARAMS['experiment'] + 'confmatrix.png')

    accuracy = (cm[0,0].item() + cm[1,1].item())/ (cm[0,0].item() + cm[1,1].item() + cm[0,1].item() + cm[1,0].item())
    specificity = cm[0,0].item()/( cm[1,0].item() + cm[0,0].item())
    recall = cm[1,1].item()/( cm[1,1].item() + cm[0,1].item())

    metrics = {
        "Accuracy":accuracy,
        "Recall": recall,
        "Specificity": specificity,
    }

    file = open('../experiments/' + PARAMS['experiment'] + 'Scores.pkl')
    pickle.dump(metrics, file)
    file.close()

    # TODO: create plots for compare loss between anomaly, and normal data.