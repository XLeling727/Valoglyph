import streamlit as st
import requests
from data import pred
import pandas as pd
import matplotlib.pyplot as plt

# Base URL for the vlresports API
BASE_URL = "http://localhost:5000/api/v1"

def get_events():
    url = f"{BASE_URL}/events"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK':
            return data['data']
        else:
            st.error(f"API returned unexpected status: {data['status']}")
            return []
    except requests.RequestException as e:
        st.error(f"Failed to fetch events: {str(e)}")
        return []


def get_matches():
    url = f"{BASE_URL}/matches"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK':
            return data['data']
        else:
            st.error(f"API returned unexpected status: {data['status']}")
            return []
    except requests.RequestException as e:
        st.error(f"Failed to fetch matches: {str(e)}")
        return []

def get_results():
    url = f"{BASE_URL}/results"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the status code is 200
        data = response.json()  # Parse the JSON response
        
        # Log the full response data for debugging
        #st.write("Full response from API:", data)
        
        # Check if the 'status' field is present and if it's 'OK'
        if 'status' in data and data['status'] == 'OK':
            return data['data']
        else:
            st.error(f"API returned unexpected status or missing 'status' field: {data}")
            return []
    except requests.RequestException as e:
        st.error(f"Failed to fetch results: {str(e)}")
        return []
    

def display_clean_results(results):
    if results:
        for match in results:
            # Extract match details
            teams = match['teams']
            team1 = teams[0]['name']
            team1_score = teams[0]['score']
            team1_country = teams[0]['country']
            team1_won = teams[0]['won']

            team2 = teams[1]['name']
            team2_score = teams[1]['score']
            team2_country = teams[1]['country']
            team2_won = teams[1]['won']

            match_event = match['event']
            match_tournament = match['tournament']
            match_status = match['status']
            match_time_ago = match['ago']
            match_image = match['img']

            # Display the match info with some formatting
            st.markdown(f"### {team1} vs {team2}")
            st.image(match_image, width=150)
            st.write(f"**Tournament**: {match_tournament}")
            st.write(f"**Event**: {match_event}")
            st.write(f"**Status**: {match_status}")
            st.write(f"**Played**: {match_time_ago} ago")
            
            # Team details and scores
            st.write(f"**{team1}** ({team1_country}) - Score: {team1_score} {'✅' if team1_won else ''}")
            st.write(f"**{team2}** ({team2_country}) - Score: {team2_score} {'✅' if team2_won else ''}")
            st.markdown("---")

    else:
        st.write("No results available.")





def filter_events(events, status_filter='all'):
    return [event for event in events if status_filter == 'all' or event['status'].lower() == status_filter]

def get_match_options(matches, results, event_name, include_results=False):
    all_matches = []
    
    if not include_results:
        all_matches = [
            match for match in matches 
            if match['tournament'] == event_name and 
            all(team['name'] != 'TBD' for team in match['teams'])
        ]
    else:
        all_matches = [
            result for result in results
            if result['tournament'] == event_name
        ]
    
    options = []
    for match in all_matches:
        team1 = match['teams'][0]['name']
        team2 = match['teams'][1]['name']
        status = match['status']
        
        if include_results:
            score1 = match['teams'][0]['score']
            score2 = match['teams'][1]['score']
            ago = match['ago']
            option = f"{team1} {score1} - {score2} {team2} ({ago} ago)"
        elif status == 'LIVE':
            score1 = match['teams'][0].get('score', '0')
            score2 = match['teams'][1].get('score', '0')
            option = f"{team1} {score1} - {score2} {team2} (LIVE)"
        elif status == 'Upcoming':
            match_time = match.get('in', 'Time not available')
            option = f"{team1} vs {team2} (In {match_time})"
        else:
            option = f"{team1} vs {team2} ({status})"
        
        options.append((option, match))
    
    return options





def display_prediction_statistics(prediction):
    # Bar chart
    data = {
        "Metric": [
            f"{prediction[0]} Win Probability", 
            f"{prediction[4]} Win Probability", 
            "Upset Probability", 
            "Prediction Confidence"
        ],
        "Percentage": [
            prediction[1] * 100, 
            100 - (prediction[1] * 100), 
            prediction[2] * 100, 
            prediction[3] * 100
        ]
    }
    df = pd.DataFrame(data)



    # Pie chart with custom colors and black background
    labels = [f"{prediction[0]} Win Probability", f"{prediction[4]} Win Probability"]
    sizes = [prediction[1] * 100, 100 - (prediction[1] * 100)]
    colors = ['#FF0000', '#78d8bb']  # Red and #78d8bb

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')  # Set the background color of the figure to black
    ax.set_facecolor('0e1217')         # Set the background color of the axes to black
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    
    # Set the color of the labels and percentages to white
    for text in texts:
        text.set_color("white")
    for autotext in autotexts:
        autotext.set_color("white")
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

    # Table
    df["Percentage"] = df["Percentage"].apply(lambda x: f"{x:.2f}%")
    st.table(df)

  
def main():
    input = dict()
    st.title("ValoStats: Valorant Event Predictor")

    events = get_events()

    if events:
        status_filter = st.selectbox("Filter events by status", ['all', 'ongoing', 'upcoming'])
        filtered_events = filter_events(events, status_filter)
        
        event_names = [event['name'] for event in filtered_events]
        selected_event = st.selectbox("Select an event", [""] + event_names)
        input["selected_event"] = selected_event
        
        if selected_event:
            # Ask the user whether they want Upcoming/Live Matches or Past Results
            match_type = st.radio("Select match type", ["Upcoming/Live Matches", "Past Results"])

            match_options = []

            # Fetch matches or results only when needed based on the user selection
            if match_type == "Upcoming/Live Matches":
                # Only fetch matches when "Upcoming/Live Matches" is selected
                matches = get_matches()
                match_options = get_match_options(matches, [], selected_event, include_results=False)
                
            elif match_type == "Past Results":
                # Only fetch results when "Past Results" is selected
                results = get_results()
                display_clean_results(results)
                match_options = get_match_options([], results, selected_event, include_results=True)

            if match_options:
                match_descriptions = ["Select a match"] + [option[0] for option in match_options]
                selected_match_index = st.selectbox("Select a match for prediction", range(len(match_descriptions)), format_func=lambda x: match_descriptions[x])

                if selected_match_index > 0:
                    selected_match = match_options[selected_match_index - 1][1]

                    # Map selection
                    maps = ["Bind", "Haven", "Split", "Ascent", "Icebox", "Breeze", "Fracture", "Abyss", "Lotus", "Sunset", "Pearl"]
                    selected_map = st.selectbox("Select the map", maps, index=None)

                    if selected_map:
                        st.write(f"Selected match: {match_descriptions[selected_match_index]}")
                        st.write(f"Selected map: {selected_map}")

                        # Updated logic to handle both "vs" and "LIVE" match descriptions
                        match_description = match_descriptions[selected_match_index]

                        if "vs" in match_description:
                            # For standard matches like "Team A vs Team B (In X time)"
                            split1 = match_description.index("vs")
                            split2 = match_description.index("(In")
                            team_a = match_description[0:split1-1].strip()
                            team_b = match_description[split1+3:split2-1].strip()
                        elif "-" in match_description and "(LIVE)" in match_description:
                            # For ongoing matches like "Team A 0 - 0 Team B (LIVE)"
                            split1 = match_description.index("-")
                            split2 = match_description.index("(LIVE")
                            team_a = match_description[:split1].strip().split()[0]
                            team_b = match_description[split1+1:split2].strip().split()[1]
                        else:
                            st.error("Unknown match format.")
                            return

                        # Proceed with prediction after extracting the teams
                        prediction = pred(selected_map, team_a, team_b, 200, 200)

                        print(f"{prediction[0]} has a {prediction[1] * 100:.2f}% chance of winning this map.")
                        print(f"{prediction[4]} has a {100 - (prediction[1] * 100):.2f}% chance of winning this map.")
                        print(f"There is an {prediction[2] * 100:.2f}% chance of an upset.")
                        print(f"This prediction was made with an {prediction[3] * 100:.2f}% confidence") 

                        st.write(f"Bet {prediction[5] * (1-prediction[2]):.2f} on {prediction[6]}.")
                        st.write(f"Bet {prediction[5] * (prediction[2]):.2f} on {prediction[0] if prediction[6] != prediction[0] else prediction[4]}.")

                        display_prediction_statistics(prediction)
            else:
                st.write("No matches available for this event.")
    else:
        st.error("No events data available. Please try again later.")



if __name__ == "__main__":
    main()

# Sidebar description
st.sidebar.markdown(
    "<h1 style='color: red; font-size: 36px; margin-top: 30px;'>About ValoStats</h1>",
    unsafe_allow_html=True
)
st.sidebar.write("""
ValoStats is your go-to tool for mastering Valorant.
""")