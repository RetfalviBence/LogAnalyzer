from LogAnalyzer.logparsing import Drain

""" Script for starting the logParsing in HDFS, have to change params in other dataset """

path = '' # add the path of the log file
removeCol = [0, 1]
st = 0.2
depth = 4
rex = ['blk_(|-)[0-9]+','(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)']
logName = 'HDFS.log'

parserPara = Drain.Para(path=path, st=st, removeCol=removeCol, rex=rex, depth=depth, logName=logName)	
myParser = Drain.Drain(parserPara)
myParser.mainProcess()