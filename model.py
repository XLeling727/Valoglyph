import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  
from enum import Enum

with open("Dataset/overview.csv", "r") as f:
    lines = f.readlines()
with open("Dataset/overview.csv", "w") as f:
    for line in lines:
        if "attack" not in line.strip("\n") and "defend" not in line.strip("\n"):
            f.write(line)

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

data1 = np.genfromtxt("Dataset/maps_scores.csv", delimiter=',', dtype=None)

score = data1[1:, [6, 11, 5]]

# map, player, team, agent, ACE, winner
XY = data2[1:, [4, 5, 6, 7, 8, 0]]

for i in range(0, score.size() // 3):
        if score[i, 0] < score[i, 1]:
                score[i, 2] = data1[i+1, 10]

indata = np.repeat(score, repeats=5, axis=0)

print(indata)

XY[1:, 5] = indata[1:, 3]

print(XY) # Input for training
