import pandas as pd
import numpy as np

"""
Script for create one hot encoded form of log messages from preprocessed drain output .cs files, and HDFS anomaly labels
TODO: change keras Tokenizer, to selfmade, or pytorch implementation
"""


print("start")
BLOCK_LABEL = "" # labels
EVENT_LOG =  "" #event_log .csv
LOG_BLOCK = "" #log_block .csv
EVENT_TEMPLATE = "" # preprocessed events from drain (IMPORTANT to rewrite all log message word to interpretable english word)
OUTPUTPATH = "" # path folder for output data (logdata.npy, loglabel.npy files will be created there)

eventFile = open(EVENT_TEMPLATE, 'r')
Lines = eventFile.readlines()
eventList = []
for line in Lines:
    eventList.append(line.strip())

eventVectors = dict()
eventID=0
for event in eventList:
    oneHotArray = np.zeros(len(eventList))
    oneHotArray[eventID] = 1
    eventVectors[eventID] = oneHotArray
    eventID = eventID + 1

# Create dataset
print("create dataet")
x = []
y = []
counter = 0
for blckId, label in zip(BLOCK_LABEL['BlockId'],BLOCK_LABEL['Label']):
    counter = counter+1
    if counter%100 == 0:
        print(counter)
    y.append(label)
    actBlockEmbedding = []
    logList = LOG_BLOCK[LOG_BLOCK['blockID'] == blckId]['logID']
    for log in logList:
        eventID = EVENT_LOG[EVENT_LOG['logID'] == log]['eventID'].values[0]
        actBlockEmbedding.append(eventVectors[eventID])
    x.append(actBlockEmbedding)

a = np.asarray(x)
np.save(OUTPUTPATH + "logdataOHE.npy", a)
b = np.asarray(y)
np.save(OUTPUTPATH + "loglabelOHE.npy", b)