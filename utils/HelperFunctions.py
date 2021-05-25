import torch
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def getDevice(gpu_num):
    if torch.cuda.is_available():  
        dev = "cuda:" + gpu_num
    else:  
        dev = "cpu"  
    return torch.device(dev)

def getBestCut(tpr, fpr, tresholds):
    diff = tpr-fpr
    diff = diff.tolist()
    v = max(diff)
    best_cut = diff.index(v)

    return tresholds[best_cut]

def confusion_matrix(preds, labels):
    matrix = np.zeros((2,2))
#    _, preds = torch.max(outputs, dim=1)
    for pred, label in zip(preds, labels):
        matrix[pred, label] +=1
    return matrix