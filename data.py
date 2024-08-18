# %%
import pandas as pd
import numpy as np

# %%
TEAMS = {"MIBR": 0, 
         "Leviatán": 1, 
         "Sentinels": 2, 
         "NRG Esports": 3, 
         "FURIA": 4, 
         "100 Thieves": 5, 
         "LOUD" : 6, 
         "Evil Geniuses" : 7, 
         "G2 Esports" : 8, 
         "Cloud9" : 9, 
         "KRÜ Esports" : 10, 
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

TEAMS_CAP = {}
MAPS_CAP = {}

for key in TEAMS:
    TEAMS_CAP[key.upper()] = TEAMS[key]
for key in MAPS:
    MAPS_CAP[key.upper()] = MAPS[key]

# %%
data = pd.read_csv("Dataset/maps_scores.csv")
data = data[["Map", "Team A", "Team B", "Team A Score", "Team B Score"]]
data["Team A Delta"] = data["Team A Score"] - data["Team B Score"]
data["Team B Delta"] = data["Team B Score"] - data["Team A Score"]
data.loc[data["Team A Delta"] < 0, "Team A Win Chance"] = 0
data.loc[data["Team A Delta"] > 0, "Team A Win Chance"] = 100
data

# %%
ACS = pd.read_csv("Dataset/overview.csv")
ACS = ACS[["Match Type", "Map", "Match Name", "Team", "Average Combat Score"]]
ACS.groupby("Team")['Average Combat Score']

# %%
TeamAcs = ACS.groupby(["Team", "Map"])['Average Combat Score'].mean().reset_index()
TeamAcs

# %%
merged_data = data.merge(TeamAcs, left_on=["Team A", "Map"], right_on=["Team", "Map"], how="left")
merged_data = merged_data.merge(TeamAcs, left_on=["Team B", "Map"], right_on=["Team", "Map"], how="left")
merged_data = merged_data.rename(columns={"Average Combat Score_x" : "Team A ACS", "Average Combat Score_y" : "Team B ACS"})
merged_data.drop(columns=["Team_x", "Team_y"], inplace=True)
merged_data = merged_data.replace({"Team A" : TEAMS, "Team B" : TEAMS, "Map" : MAPS})
merged_data

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib


# %%
merged_data = merged_data.drop(["Team A Score", "Team B Score", "Team A Delta", "Team B Delta"], axis=1)
X = merged_data.drop(["Team A Win Chance"], axis = 1)
X

# %%
Y = merged_data["Team A Win Chance"]
Y = Y.replace({100.0: 1, 0.0: 0})
Y = Y.astype(int)

Y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=True)
# rf.fit(X_train, y_train)
log_regr = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, multi_class='auto')
# log_regr.fit(X_train, y_train)
bag_clf = BaggingClassifier(estimator=log_regr, n_estimators=20, bootstrap=True, warm_start=True)

boost_clf = GradientBoostingClassifier(loss="log_loss",n_estimators=80 , verbose=True)
#mlp_clf = MLPClassifier()

# %%
#Boosting clf
#Uncomment to use
boost_clf.fit(X_train, y_train)
boost_clf_pred = boost_clf.predict(X_test)

accuracy = accuracy_score(y_pred=boost_clf_pred, y_true=y_test)
print("accuracy: ", accuracy)

# %%
X_test

# %%
#Getting the first 5 data points
filtered_rows = []
team = 1

for index, row in merged_data.iterrows():
    if row['Team A'] == team or row['Team B'] == team:
        filtered_rows.append(row)
    
    # Stop once we've collected 15 rows
    if len(filtered_rows) == 15:
        break

# Convert the filtered_rows list to a DataFrame
filtered_df = pd.DataFrame(filtered_rows)

# Assuming you have a column named "Team A Win Chance" in your original dataset
# Drop the "Team A Win Chance" column to create new_X
new_X = filtered_df.drop(["Team A Win Chance"], axis=1)

# Extract and transform the "Team A Win Chance" column into new_Y
new_Y = filtered_df["Team A Win Chance"]
new_Y = new_Y.replace({100.0: 1, 0.0: 0})
new_Y = new_Y.astype(int)

# Split the data into training and testing sets
new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_X, new_Y, test_size=0.2, random_state=42)

# boost_clf.fit(new_X_train, new_y_train)
# # new_pred = boost_clf.predict(X_test)
# # print(X_test, y_test)
# custom_data_df_old = pd.DataFrame([[3,1,21,200,205]], columns=new_X_train.columns)
# new_pred = boost_clf.predict(custom_data_df_old)
# print(new_pred)

#new_accuracy = accuracy_score(y_pred=new_pred, y_true=new_y_test)

#  accuracy = accuracy_score(y_pred=new_pred, y_true=new_y_test)
# print("accuracy: ", accuracy)




# %%
#Training the boosting model on the new data
boost_clf.fit(new_X, new_Y)

yyypred = boost_clf.predict(new_X_test)
new_accuracy = accuracy_score(y_pred=yyypred, y_true=new_y_test)
print("new accuracy", new_accuracy)
custom_data_df_old = pd.DataFrame([[3,1,21,200,205]], columns=new_X_train.columns)
probability_old = boost_clf.predict_proba(custom_data_df_old)[:, 1]
new_ypred = boost_clf.predict(custom_data_df_old)
print("team with prob", new_ypred, probability_old[0])

# %%
#Bagging pred
#uncomment to use 
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {np.around(accuracy*100, 2)}%")

# %%
# #y_pred = rf.predict(X_test)
# y_pred = log_regr.predict(X_test)
# print(y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# # conf1_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
# # fig, ax = plt.subplots()

# # cm_display = ConfusionMatrixDisplay(confusion_matrix=conf1_matrix, display_labels=rf.classes_)
# # cm_display.plot(ax=ax, cmap='Greens')

# print(f"Accuracy: {np.around(accuracy*100, 2)}%")
# # print("Classification Report:")
# # print(report)

# %%
random_row = X_train.sample(n=1)
# custom_data = np.array([[8, 10, 5, 13, 11, 2]])
custom_data_df = pd.DataFrame([[3,1,21,200,205]], columns=X_train.columns)
probability_y1 = boost_clf.predict_proba(custom_data_df)[:, 1]
rand_pred = boost_clf.predict(custom_data_df)
loss = 1-  probability_y1
if rand_pred == 0:
    print(f"Team A will lose with a probability of: { np.around(loss*100, 2)}%")
else:
    print(f"Team A will win with a probability of:{np.around(probability_y1*100, 2)}%")


# %%
correlation_matrix = merged_data.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# %%
#Complete logistic for all probability and recent 5 games probability before calling this function
def calculate_upset_probability(season_prob, recent_prob, weight_recent=0.34, num_simulations=100000):
    """
    Calculate the probability of an upset and the confidence in the prediction.
    
    :param season_prob: Probability of Team A winning based on season-long data
    :param recent_prob: Probability of Team A winning based on recent 5 matches
    :param weight_recent: Weight given to recent performance (0 to 1)
    :param num_simulations: Number of Monte Carlo simulations to run
    :return: Tuple of (upset_probability, confidence)
    """
    # Combine probabilities with weighted average
    combined_prob = (1 - weight_recent) * season_prob + weight_recent * recent_prob
    
    # Calculate parameters for Beta distribution
    alpha = combined_prob * 100
    beta_param = (1 - combined_prob) * 100
     
    # Run Monte Carlo simulation
    simulations = np.random.beta(alpha, beta_param, num_simulations)
    
    # Calculate upset probability
    upset_probability = np.mean(simulations < 0.5)
    
    # Calculate confidence (as the proportion of simulations within ±10% of the mean)
    confidence = np.mean((combined_prob - 0.1 < simulations) & (simulations < combined_prob + 0.1))
    
    return upset_probability, confidence


# %%
upset_prob, confidence = calculate_upset_probability(probability_old[0], probability_y1[0])
print(f"Combined probability of Team A winning: {(probability_old[0] + probability_y1[0]) / 2:.2f}")
print(f"Probability of an upset: {upset_prob:.2f}")
print(f"Confidence in prediction: {confidence:.2f}")

# %%
joblib.dump(boost_clf, 'boost_clf_model.pkl')

# %%
def pred(map, team_a, team_b, teama_acs, teamb_acs):

    team_a_name = team_a
    team_b_name = team_b

    map = MAPS_CAP[map.upper()]
    team_a = TEAMS_CAP[team_a.upper()]
    team_b = TEAMS_CAP[team_b.upper()]

    print("DEBUG", map, team_a, team_b)

    team = 1
    winning_team = None
    custom_data_all = pd.DataFrame([[map, team_a, team_b, teama_acs, teamb_acs]], columns=X_train.columns)
    boost_clf = GradientBoostingClassifier(loss="log_loss",n_estimators=80 , verbose=True)
    boost_clf.fit(X_train, y_train)
    boost_clf_pred = boost_clf.predict(custom_data_all)
    #accuracy = accuracy_score(y_pred=boost_clf_pred, y_true=y_test)
    #print("accuracy: ", accuracy)


    probability_all = boost_clf.predict_proba(custom_data_all)[:, 1]
    loss = 1-  probability_all
    if boost_clf_pred == 0:
        print(team_b_name, "will win with a probability of: ", probability_all)
        team = team_b
        #print(f"Team A will lose with a probability of: { np.around(loss*100, 2)}%")
    else:
        print(team_a_name, "will win with a probability of: ", probability_all)
        team = team_a

    
    #Getting the first 5 data points
    filtered_rows = []

    for index, row in merged_data.iterrows():
        if row['Team A'] == team or row['Team B'] == team:
            filtered_rows.append(row)
        
        # Stop once we've collected 15 rows
        if len(filtered_rows) == 15:
            break

    # Convert the filtered_rows list to a DataFrame
    filtered_df = pd.DataFrame(filtered_rows)

    # Assuming you have a column named "Team A Win Chance" in your original dataset
    # Drop the "Team A Win Chance" column to create new_X
    new_X = filtered_df.drop(["Team A Win Chance"], axis=1)

    # Extract and transform the "Team A Win Chance" column into new_Y
    new_Y = filtered_df["Team A Win Chance"]
    new_Y = new_Y.replace({100.0: 1, 0.0: 0})
    new_Y = new_Y.astype(int)

    # Split the data into training and testing sets
    new_X_train, _, new_y_train, _ = train_test_split(new_X, new_Y, test_size=0.2, random_state=42)
    boost_clf_last15 = GradientBoostingClassifier(loss="log_loss",n_estimators=80 , verbose=True)
    boost_clf_last15.fit(new_X_train, new_y_train)
    #Cutom data from user
    custom_data_df_last15 = pd.DataFrame([[map, team_a, team_b, teama_acs, teamb_acs]], columns=new_X_train.columns)
    
    probability_last15 = boost_clf_last15.predict_proba(custom_data_df_last15)[:, 1]
    new_ypred15 = boost_clf_last15.predict(custom_data_df_last15)
    #print("team with prob", new_ypred15, probability_last15[0])
    if new_ypred15 == 0:
        print(team_b_name, "will win with a probability of: ", probability_last15)
        #print(f"Team A will lose with a probability of: { np.around(loss*100, 2)}%")
        winning_team = team_b_name
    else:
        #print(TEAMS[team_a], "will win with a probability of: ", probability_all)
        winning_team = team_a_name  

    upset_prob, confidence = calculate_upset_probability(probability_all[0], probability_last15[0])
    print(f"Combined probability of Team A winning: {(probability_all[0] + probability_last15[0]) / 2:.2f}")
    print(f"Probability of an upset: {upset_prob:.2f}")
    print(f"Confidence in prediction: {confidence:.2f}")

    return [team_a_name, (probability_all[0] + probability_last15[0]) / 2, upset_prob, confidence,
            team_b_name]
    
    
    
    


