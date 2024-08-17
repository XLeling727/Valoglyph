import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  
from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from constants import MAPS, TEAMS

data1 = np.genfromtxt("maps_scores.csv", delimiter=',', dtype=None)
data = data1[1:, 4:15]

indata = data1[1:, [4, 5, 10]]

new_indata = np.copy(indata)
for k, v in MAPS.items(): new_indata[indata[:,0]==k, 0] = v
for k, v in TEAMS.items(): new_indata[indata[:,1]==k, 1] = v
for k, v in TEAMS.items(): new_indata[indata[:,2]==k, 2] = v

outdata = np.zeros((data.size // 11, 2))

# new_indata[:, 3] = np.abs(data[:, 2].astype(np.int64) - data[:, 7].astype(np.int64))
outdata[data[:, 2].astype(np.int64) > data[:, 7].astype(np.int64), 0] = 100
outdata[data[:, 2].astype(np.int64) < data[:, 7].astype(np.int64), 1] = 100

# print(new_indata) # Input for training
# print(outdata) # Desired output for training

# X = new_indata[0]
# x_train, x_test, y_train, y_test = train_test_split
X = []
wining_teams = []
counter = 0
testStat = np.array([[1, 8, 8]])[:, np.newaxis]
#print(new_indata[0][0], new_indata[0][1])
counter = 0
print(outdata[0][0])
while counter < len(outdata):
        if outdata[counter][0] == 100:
                #print(new_indata[counter][0])
                #new_indata[counter] = np.append(new_indata[counter],new_indata[counter][0])
                wining_teams.append(new_indata[counter][0])
        elif outdata[counter][1] == 100:
                #print(new_indata[counter[1]])
                wining_teams.append(new_indata[counter][1])
        counter += 1
print(wining_teams)
#for teams in wining_teams:
counter1 = 0
for teams in new_indata:
         teams = np.append(teams, wining_teams[counter1])
         counter1 += 1
print(new_indata)
log_regr = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, multi_class='auto')
#log_regr.fit(x_train, y_train)