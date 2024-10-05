# %%
import pandas as pd
import numpy as np
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# %%
TEAMS = {"MIBR": 1, 
         "Leviatán": 2, 
         "Sentinels": 3, 
         "NRG Esports": 4, 
         "FURIA": 5, 
         "100 Thieves": 6, 
         "LOUD" : 7, 
         "Evil Geniuses" : 8, 
         "G2 Esports" : 9, 
         "Cloud9" : 10, 
         "KRÜ Esports" : 11, 
         "Titan Esports Club" : 12, 
         "JDG Esports" : 13, 
         "All Gamers" : 14, 
         "TYLOO" : 15, 
         "Bilibili Gaming" : 16, 
         "Wolves Esports" : 17, 
         "Dragon Ranger Gaming" : 18, 
         "Nova Esports" : 19, 
         "FunPlus Phoenix" : 20, 
         "Trace Esports" : 21, 
         "EDward Gaming" : 22, 
         "ZETA DIVISION" : 23, 
         "DRX" : 24, 
         "Team Secret" : 25, 
         "BLEED" : 26, 
         "T1" : 27, 
         "Gen.G" : 28, 
         "Paper Rex" : 29, 
         "Talon Esports" : 30, 
         "Rex Regum Qeon" : 31, 
         "DetonatioN FocusMe" : 32, 
         "Global Esports" : 33, 
         "FUT Esports" : 34, 
         "KOI" : 35, 
         "BBL Esports" : 36, 
         "FNATIC" : 37, 
         "Team Heretics" : 38, 
         "Natus Vincere" : 39, 
         "GIANTX" : 40, 
         "Gentle Mates" : 41, 
         "Team Vitality" : 42, 
         "Team Liquid" : 43, 
         "Karmine Corp": 44}

MAPS = {"Bind" : 1, 
        "Haven" : 2, 
        "Split" : 3, 
        "Ascent" : 4,
        "Icebox" : 5, 
        "Breeze" : 6, 
        "Fracture" : 7, 
        "Abyss" : 8, 
        "Lotus" : 9, 
        "Sunset" : 10, 
        "Pearl" : 11}

TEAMS_CAP = {}
MAPS_CAP = {}

for key in TEAMS:
    TEAMS_CAP[key.upper()] = TEAMS[key]
for key in MAPS:
    MAPS_CAP[key.upper()] = MAPS[key]

def download_kaggle_files():
    # Initialize the API
    api = KaggleApi()
    api.authenticate()

    
    dataset_path = 'ryanluong1/valorant-champion-tour-2021-2023-data' 

    # Specify the location where you want to store the dataset files
    download_path = 'Dataset'  

    # Ensure the directory exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Specify the exact files you want to download (replace with actual filenames)
    files_to_download = ['vct_2024/matches/map_scores.csv', 'vct_2023/matches/map_scores.csv', 'vct_2024/matches/scores.csv', 'vct_2024/macthes/overview.csv']

    # Download specific files
    for file_name in files_to_download:
        print(f"Downloading {file_name} from Kaggle dataset {dataset_path}...")
        api.dataset_download_file(dataset_path, file_name, path=download_path)

    # Optionally, read and return the downloaded datasets as pandas DataFrames
    dataframes = {}
    for file_name in files_to_download:
        file_path = os.path.join(download_path, file_name)
        if os.path.exists(file_path):
            dataframes[file_name] = pd.read_csv(file_path)
            print(f"Loaded {file_name} into a DataFrame.")
        else:
            print(f"File {file_name} not found!")

    return dataframes  # Return the loaded dataframes for further use

# %%
data = pd.read_csv("Dataset/maps_scores.csv")
data = data[["Map", "Team A", "Team B", "Team A Score", "Team B Score"]]
data["Team A Delta"] = data["Team A Score"] - data["Team B Score"]
data["Team B Delta"] = data["Team B Score"] - data["Team A Score"]
data.loc[data["Team A Delta"] < 0, "Team A Win Chance"] = 0
data.loc[data["Team A Delta"] > 0, "Team A Win Chance"] = 100

def update_teams_dict(team_name):
    """Adds missing teams to the TEAMS dictionary and updates TEAMS_CAP."""
    if team_name not in TEAMS:
        # Add new team with the next available index
        new_index = max(TEAMS.values()) + 1
        TEAMS[team_name] = new_index
        TEAMS_CAP[team_name.upper()] = new_index  # Update TEAMS_CAP with uppercase team name
        print(f"Added new team: {team_name} with index {new_index}")


def check_and_update_teams_in_data(data):
    """Checks for missing teams in the dataset and updates the TEAMS dictionary."""
    for team_a, team_b in zip(data['Team A'], data['Team B']):
        update_teams_dict(team_a)
        update_teams_dict(team_b)
check_and_update_teams_in_data(data)
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
    # Add missing teams to the TEAMS and TEAMS_CAP dictionaries if not present
    update_teams_dict(team_a)
    update_teams_dict(team_b)

    # Now you can safely map the team names to their IDs
    team_a_name = team_a
    team_b_name = team_b

    map = MAPS_CAP.get(map.upper(), None)
    team_a = TEAMS_CAP.get(team_a.upper(), None)
    team_b = TEAMS_CAP.get(team_b.upper(), None)

    if map is None or team_a is None or team_b is None:
        print("Error: One of the teams or map is not in the dictionary. Returning 50/50 prediction.")
        # If the map or teams are missing, return a 50/50 prediction
        return [team_a_name, 0.5, 0.5, 1.0, team_b_name, 0.0, "Unknown"]

    print("DEBUG", map, team_a, team_b)

    team = 1
    winning_team = None
    custom_data_all = pd.DataFrame([[map, team_a, team_b, teama_acs, teamb_acs]], columns=X_train.columns)
    boost_clf = GradientBoostingClassifier(loss="log_loss", n_estimators=80, verbose=True)
    boost_clf.fit(X_train, y_train)
    boost_clf_pred = boost_clf.predict(custom_data_all)

    probability_all = boost_clf.predict_proba(custom_data_all)[:, 1]
    loss = 1 - probability_all
    if boost_clf_pred == 0:
        print(f"{team_b_name} will win with a probability of {probability_all}")
        team = team_b
    else:
        print(f"{team_a_name} will win with a probability of {probability_all}")
        team = team_a

    # Getting the first 15 data points for the selected teams
    filtered_rows = []
    for index, row in merged_data.iterrows():
        if row['Team A'] == team or row['Team B'] == team:
            filtered_rows.append(row)

        # Stop once we've collected 15 rows
        if len(filtered_rows) == 15:
            break

    # Convert the filtered_rows list to a DataFrame
    filtered_df = pd.DataFrame(filtered_rows)

    # Check if "Team A Win Chance" exists before dropping
    if "Team A Win Chance" in filtered_df.columns:
        new_X = filtered_df.drop(["Team A Win Chance"], axis=1)
    else:
        new_X = filtered_df  # If the column doesn't exist, no need to drop

    # Check if new_X has columns before proceeding
    if new_X.empty or len(new_X.columns) == 0:
        print("Error: new_X DataFrame is empty or has no columns. Returning 50/50 prediction.")
        return [team_a_name, 0.5, 0.5, 1.0, team_b_name, 0.0, "Unknown"]

    # Extract and transform the "Team A Win Chance" column into new_Y if it exists
    if "Team A Win Chance" in filtered_df.columns:
        new_Y = filtered_df["Team A Win Chance"].replace({100.0: 1, 0.0: 0}).astype(int)
    else:
        new_Y = pd.Series([])  # Handle missing labels if necessary

    # Split the data into training and testing sets
    if not new_Y.empty:
        new_X_train, _, new_y_train, _ = train_test_split(new_X, new_Y, test_size=0.2, random_state=42)
        boost_clf_last15 = GradientBoostingClassifier(loss="log_loss", n_estimators=80, verbose=True)
        boost_clf_last15.fit(new_X_train, new_y_train)
    else:
        boost_clf_last15 = boost_clf  # Fallback to the original model

    # Custom data prediction (ensure new_X has valid columns)
    custom_data_df_last15 = pd.DataFrame([[map, team_a, team_b, teama_acs, teamb_acs]], columns=new_X.columns)
    probability_last15 = boost_clf_last15.predict_proba(custom_data_df_last15)[:, 1]
    new_ypred15 = boost_clf_last15.predict(custom_data_df_last15)

    if new_ypred15 == 0:
        print(f"{team_b_name} will win with a probability of {probability_last15[0]}")
        winning_team = team_b_name
    else:
        print(f"{team_a_name} will win with a probability of {probability_last15[0]}")
        winning_team = team_a_name

    # Upset probability and confidence
    upset_prob, confidence = calculate_upset_probability(probability_all[0], probability_last15[0])
    print(f"Combined probability of Team A winning: {(probability_all[0] + probability_last15[0]) / 2:.2f}")
    print(f"Probability of an upset: {upset_prob:.2f}")
    print(f"Confidence in prediction: {confidence:.2f}")

    # Betting logic
    max_value = 1200000
    if ((probability_all[0] + probability_last15[0]) / 2) < 0.5:
        if upset_prob > 0.5:
            bet_amount = max_value * upset_prob * confidence
            print(f"Bet {bet_amount:.2f} on {winning_team} as an upset.")
        else:
            print(f"Do not bet; probability and upset risk are too low.")
    else:
        bet_amount = max_value * ((probability_all[0] + probability_last15[0]) / 2) * confidence
        print(f"Bet {bet_amount:.2f} on {winning_team}.")

    return [team_a_name, (probability_all[0] + probability_last15[0]) / 2, upset_prob, confidence, team_b_name, bet_amount, winning_team]



    
    
    
    


