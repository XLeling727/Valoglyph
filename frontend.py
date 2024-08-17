import streamlit as st
import requests

def get_events():
    url = "https://vlr.orlandomm.net/api/v1/events"
    params = {"page": 1, "status": "all", "region": "all"}
    try:
        response = requests.get(url, params=params)
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
    url = "https://vlr.orlandomm.net/api/v1/matches"
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

def filter_events(events, status_filter='all'):
    return [event for event in events if status_filter == 'all' or event['status'].lower() == status_filter]

def display_matches(matches, event_name):
    confirmed_matches = [
        match for match in matches 
        if match['tournament'] == event_name and 
        all(team['name'] != 'TBD' for team in match['teams'])
    ]
    
    for match in confirmed_matches:
        team1 = match['teams'][0]['name']
        team2 = match['teams'][1]['name']
        status = match['status']
        
        if status == 'LIVE':
            score1 = match['teams'][0].get('score', '0')
            score2 = match['teams'][1].get('score', '0')
            st.write(f"{team1} {score1} - {score2} {team2} (LIVE)")
        elif status == 'Upcoming':
            match_time = match.get('in', 'Time not available')
            st.write(f"{team1} vs {team2} (In {match_time})")
        else:
            st.write(f"{team1} vs {team2} ({status})")

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
            display_matches(matches, selected_event)
            
            # Map selection
            maps = ["Haven", "Split", "Ascent", "Icebox", "Breeze", "Fracture", "Abyss", "Lotus", "Sunset", "Pearl"]
            selected_map = st.selectbox("Select the map", maps, index=None)

            if selected_map:
                st.write(f"Selected map: {selected_map}")
                # Here you can add your prediction logic
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