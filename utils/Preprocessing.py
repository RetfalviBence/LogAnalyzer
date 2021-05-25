import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import seaborn as sns; sns.set_theme()
from sklearn.utils import shuffle

# Perprocessing methods For HDFS dataset
class LogDataset(Dataset):
    def __init__(self, data, target, device="cpu"):
        self.data = nn.utils.rnn.pad_sequence(data, batch_first = True)
        self.target = target.squeeze()
        self.lengths = torch.tensor([x.size(0) for x in data])
        self.device = device

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx].to(self.device), self.lengths[idx], self.target[idx].to(self.device)



def preprocessData(data, label, val_anomaly_samples, val_normal_samples, train_anomaly_samples, train_nomral_samples, val_device="cpu", train_device="cpu"):
    # data pereprocess
    lb = preprocessing.LabelBinarizer()
    
    # convert text labels to binary
    labelNum = lb.fit_transform(label)
    labelNum = labelNum.astype('int64')

    anomaly_mask = labelNum == 0
    normal_mask = labelNum == 1
    
    # select anomalies
    anomaly = data[anomaly_mask.squeeze()]
    np.random.shuffle(anomaly)
    val_data_anomaly = anomaly[0:val_anomaly_samples]
    val_target_anomaly = np.ones((val_anomaly_samples,1), dtype= np.int64)

    # select normals
    normal  = data[normal_mask.squeeze()]
    np.random.shuffle(normal)
    val_data_normal = normal[0:val_normal_samples]
    val_target_normal = np.zeros((val_normal_samples,1), dtype= np.int64)
    
    # concatanate data
    val_data= np.append(val_data_normal, val_data_anomaly)
    val_target = np.append(val_target_normal, val_target_anomaly)

    # create validation data, with preprocessing.
    val_datanp = []
    for element in val_data:
        scaledData = preprocessing.StandardScaler().fit_transform(np.array(element))
        val_datanp.append(torch.from_numpy(scaledData))


    # select normals for train
    train_data_normal = normal[val_normal_samples:( val_normal_samples+train_nomral_samples )]
    train_target_normal = np.zeros((train_nomral_samples, 1), dtype= np.int64)
    
    # select anomalies for train
    train_data_anomaly = np.array([])
    if train_anomaly_samples > 0:
        train_data_anomaly = anomaly[val_anomaly_samples:(val_anomaly_samples+train_anomaly_samples)]
        train_target_anomaly = np.ones((train_anomaly_samples,1), dtype= np.int64)

    # concatanate data
    train_data = np.append(train_data_normal, train_data_anomaly)
    Y = np.append(train_target_normal, train_target_anomaly)
    train_target = torch.from_numpy(Y)
    train_data, train_target = shuffle(train_data, train_target, random_state=0)
    
    # create train data, with preprocessing.
    train_datanp = []
    for element in train_data:
        scaledData = preprocessing.StandardScaler().fit_transform(np.array(element))
        train_datanp.append(torch.from_numpy(scaledData))


    data_set_train = LogDataset(train_datanp, train_target, train_device)
    data_set_val = LogDataset(val_datanp, val_target, val_device)
    train_loader = DataLoader(data_set_train, 1, shuffle=True)
    val_loader = DataLoader(data_set_val, val_data, 1)

    return val_loader, train_loader