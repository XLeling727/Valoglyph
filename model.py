import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  
from enum import Enum

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

print(new_indata) # Input for training
print(outdata) # Desired output for training


testStat = np.array([[1, 8, 8]])[:, np.newaxis]