import streamlit as st
import requests

# Base URL for the vlresports API
BASE_URL = "https://vlr.orlandomm.net/api/v1"

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
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK':
            return data['data']
        else:
            st.error(f"API returned unexpected status: {data['status']}")
            return []
    except requests.RequestException as e:
        st.error(f"Failed to fetch results: {str(e)}")
        return []

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

def main():
    st.title("ValoStats: Valorant Event Predictor")

    events = get_events()
    
    if events:
        status_filter = st.selectbox("Filter events by status", ['all', 'ongoing', 'upcoming'])
        filtered_events = filter_events(events, status_filter)
        
        event_names = [event['name'] for event in filtered_events]
        selected_event = st.selectbox("Select an event", [""] + event_names)

        if selected_event:
            matches = get_matches()
            results = get_results()
            
            match_type = st.radio("Select match type", ["Upcoming/Live Matches", "Past Results"])
            
            if match_type == "Upcoming/Live Matches":
                match_options = get_match_options(matches, results, selected_event, include_results=False)
            else:
                match_options = get_match_options(matches, results, selected_event, include_results=True)
            
            if match_options:
                match_descriptions = ["Select a match"] + [option[0] for option in match_options]
                selected_match_index = st.selectbox("Select a match for prediction", range(len(match_descriptions)), format_func=lambda x: match_descriptions[x])
                
                if selected_match_index > 0:
                    selected_match = match_options[selected_match_index - 1][1]
                    
                    # Map selection
                    maps = ["Haven", "Split", "Ascent", "Icebox", "Breeze", "Fracture", "Abyss", "Lotus", "Sunset", "Pearl"]
                    selected_map = st.selectbox("Select the map", maps, index=None)

                    if selected_map:
                        st.write(f"Selected match: {match_descriptions[selected_match_index]}")
                        st.write(f"Selected map: {selected_map}")
                        # Here you can add your prediction logic
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