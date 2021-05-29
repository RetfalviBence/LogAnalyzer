import os
import pandas as pd
import re

""" 

Script for create .csv files from HDFS data, and Drain output.

This .csv files will contain block_id - log_id, log_id - event_id pairs
TODO: create relational database for store this data

"""

LOD_ID_DRAIN = '' # Path for the folder with drain output log_id files
EVENT_ID_LOG_ID_PATH = "" # file name with path to output .csv files, that will contain event_id log_id pairs
DATA_PATH = "" # HDFS log data path

# get eventId with logID
eventID = 0
eventLogIdParse = []
for templateFile in os.listdir(LOD_ID_DRAIN):
    theFile = open(LOD_ID_DRAIN + templateFile)
    lines = theFile.readlines()
    for line in lines:
        logID = int(line.strip())
        logDict = {"logID":logID, "eventID":eventID}
        eventLogIdParse.append(logDict)
    eventID = eventID + 1
dfLogEvent = pd.DataFrame.from_dict(eventLogIdParse)
dfLogEvent.to_csv(EVENT_ID_LOG_ID_PATH)

# get block ids with logID
logFile = open('../../DataFiles/HDFS/rawLog/HDFS.log')
Lines = logFile.readlines()

logList = []
rex = 'blk_(|-)[0-9]+'
for line in Lines:
    splittedLine = line.split(maxsplit = 3)
    idString = splittedLine[0] + splittedLine[1] + splittedLine[2]
    logID = int(idString)
    block = re.search('blk_(|-)[0-9]+', line)
    blockId = block.group()
    logRow = {"logID": logID, "blockID": blockId}
    logList.append(logRow)

dfLogBlock = pd.DataFrame.from_dict(logList)
dfLogEvent.to_csv(DATA_PATH)