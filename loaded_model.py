import pickle
import pandas as pd

def prediction():
    X_test =  {
            "selectedevent":"Valorant Champions 2024",
            "Map":"Breeze",
            "Team A":"Sentinels",
            "Team B":"FNATIC",
            "Team A ACS":200,
            "Team B ACS":200
            }

    with open('/Users/varunsingh/Desktop/Valostats/boostclfmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    print(f"Model type: {type(model)}")
    X_pd = pd.DataFrame([X_test])
    print (X_pd)
    predictions = model.predict(X_pd)
    print (predictions)
    return predictions





if __name__ == "__main":
    prediction()