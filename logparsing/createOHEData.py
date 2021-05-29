import pandas as pd
import numpy as np

print("start")
BLOCK_LABEL = pd.read_csv('anomaly_label.csv')
EVENT_LOG = pd.read_csv('EVENTID_LOGID.csv')
LOG_BLOCK = pd.read_csv('LOGID_BLOCK.csv')

eventFile = open('logTemplates.txt', 'r')
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
np.save("logdataOHE.npy", a)
b = np.asarray(y)
np.save("loglabelOHE.npy", b)