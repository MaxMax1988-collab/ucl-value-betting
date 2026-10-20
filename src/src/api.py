import requests
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_ucl_odds(api_key):
    """Pobiera mecze i kursy dla Ligi Mistrzów z The Odds API."""
    if not api_key or api_key == "TWOJ_KLUCZ_API":
        return get_mock_matches()

    url = "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/odds/"
    params = {
        'apiKey': api_key,
        'regions': 'eu',
        'markets': 'h2h',
        'dateFormat': 'iso'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            matches = []
            for match in data:
                home_team = match.get('home_team')
                away_team = match.get('away_team')
                bookmakers = match.get('bookmakers', [])
                odds = {'1': None, 'X': None, '2': None}
                
                if bookmakers:
                    h2h_market = next((m for m in bookmakers[0]['markets'] if m['key'] == 'h2h'), None)
                    if h2h_market:
                        for outcome in h2h_market['outcomes']:
                            if outcome['name'] == home_team:
                                odds['1'] = outcome['price']
                            elif outcome['name'] == away_team:
                                odds['2'] = outcome['price']
                            elif outcome['name'] == 'Draw':
                                odds['X'] = outcome['price']

                matches.append({
                    "id": match.get('id'),
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": match.get('commence_time'),
                    "odds": odds
                })
            return matches
        else:
            return get_mock_matches()
    except Exception:
        return get_mock_matches()

def get_mock_matches():
    return [
        {
            "id": "mock_1",
            "home_team": "Real Madrid",
            "away_team": "Manchester City",
            "commence_time": "2026-11-26T20:00:00Z",
            "odds": {"1": 2.40, "X": 3.50, "2": 2.80}
        },
        {
            "id": "mock_2",
            "home_team": "FC Bayern München",
            "away_team": "FC Barcelona",
            "commence_time": "2026-11-27T20:00:00Z",
            "odds": {"1": 2.10, "X": 3.70, "2": 3.20}
        }
    ]
  
