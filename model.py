import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  
from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


TEAMS = {"MIBR": 0, 
         "LeviatÃ¡n": 1, 
         "Sentinels": 2, 
         "NRG Esports": 3, 
         "FURIA": 4, 
         "100 Thieves": 5, 
         "LOUD" : 6, 
         "Evil Geniuses" : 7, 
         "G2 Esports" : 8, 
         "Cloud9" : 9, 
         "KRÃœ Esports" : 10, 
         "Titan Esports Club" : 11, 
         "JDG Esports" : 12, 
         "All Gamers" : 13, 
         "TYLOO" : 14, 
         "Bilibili Gaming" : 15, 
         "Wolves Esports" : 16, 
         "Dragon Ranger Gaming" : 17, 
         "Nova Esports" : 18, 
         "FunPlus Phoenix" : 19, 
         "Trace Esports" : 20, 
         "EDward Gaming" : 21, 
         "ZETA DIVISION" : 22, 
         "DRX" : 23, 
         "Team Secret" : 24, 
         "BLEED" : 25, 
         "T1" : 26, 
         "Gen.G" : 27, 
         "Paper Rex" : 28, 
         "Talon Esports" : 29, 
         "Rex Regum Qeon" : 30, 
         "DetonatioN FocusMe" : 31, 
         "Global Esports" : 32, 
         "FUT Esports" : 33, 
         "KOI" : 34, 
         "BBL Esports" : 35, 
         "FNATIC" : 36, 
         "Team Heretics" : 37, 
         "Natus Vincere" : 38, 
         "GIANTX" : 39, 
         "Gentle Mates" : 40, 
         "Team Vitality" : 41, 
         "Team Liquid" : 42, 
         "Karmine Corp": 43}

MAPS = {"Bind" : 0, 
        "Haven" : 1, 
        "Split" : 2, 
        "Ascent" : 3,
        "Icebox" : 4, 
        "Breeze" : 5, 
        "Fracture" : 6, 
        "Abyss" : 7, 
        "Lotus" : 8, 
        "Sunset" : 9, 
        "Pearl" : 10}

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