# Imports
import math
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# helper functions
print('script start')
def eventToVector(event, tokenEmbeddings, wordWeights):
    # input: an event
    # output: 
    vectorList = []
    for wordToken in event:
        actWeightedVector = tokenEmbeddings[wordToken] * wordWeights[wordToken]
        vectorList.append(actWeightedVector)
    aggregatedVector = sum(vectorList)/len(event)
    return aggregatedVector

BLOCK_LABEL = pd.read_csv('anomaly_label.csv')
EVENT_LOG = pd.read_csv('EVENTID_LOGID.csv')
LOG_BLOCK = pd.read_csv('LOGID_BLOCK.csv')

# Create word embeddings

# read preprocessed log events
eventFile = open('logTemplates.txt', 'r')
Lines = eventFile.readlines()
eventList = []
for line in Lines:
    eventList.append(line.strip())

# create vocab, converts words to token
tokenizer = Tokenizer(num_words=1000, lower=True)
tokenizer.fit_on_texts(eventList)
sequences = tokenizer.texts_to_sequences(eventList)
tokenizer.texts_to_sequences_generator(sequences)

# read pretrained glove word embeddings
wordEmbeddings = dict()
gloveFile = open('glove.6B.100d.txt', encoding="utf8")

for line in gloveFile:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    wordEmbeddings[word] = coefs

# create word - embedding dict
tokenEmbeddings = dict()

for logWord in tokenizer.word_index:
    tokenEmbeddings[tokenizer.word_index[logWord]] = wordEmbeddings[logWord]


# Calculate tf - idf

# tf
countWords = dict()
tf = dict()
for event in sequences:
    for wordToken in event:
        if wordToken in countWords:
            countWords[wordToken] = countWords[wordToken] + 1
        else:
            countWords[wordToken] = 1

countAllWords = sum(list(countWords.values()))
for wordToken in countWords:
    tf[wordToken] = countWords[wordToken]/countAllWords

# idf TODO write tf to this aproach
idf = dict()
for wordIndex in tokenizer.index_word:
    #actWord = tokenizer.index_word[wordIndex]
    countWord = 0
    for event in sequences:
        if (wordIndex in event):
            countWord = countWord+1
    idf[wordIndex] = math.log(len(sequences)/countWord)

# create weights to summerize vectors
wordWegiht = dict()
for token in idf:
    wordWegiht[token] = idf[token] * tf[token]

# create event - vector
eventVectors = dict()
eventID = 0
for event in sequences:
    eventVectors[eventID] = eventToVector(event, tokenEmbeddings, wordWegiht)
    eventID = eventID +1

# Create dataset
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
np.save("logdata.npy", a)
b = np.asarray(y)
np.save("loglabel.npy", b)
