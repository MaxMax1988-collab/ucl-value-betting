import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from scipy.stats import poisson

# ==========================================
# KONFIGURACJA STRONY
# ==========================================
st.set_page_config(
    page_title="UCL Predictor & Value Betting Hub",
    page_icon="⚽",
    layout="wide"
)

ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", "99f7bfd9a3977d8d7cf04dba05b03a89")
WEIGHTS_FILE = "team_weights.json"
BETS_FILE = "bets_history.json"
PROCESSED_MATCHES_FILE = "processed_matches.json"

# ==========================================
# POMOCNICZE: PLIKI JSON I KALKULATOR EV
# ==========================================
def load_json_file(filepath, default_data):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_data
    return default_data

def save_json_file(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def calculate_ev(prob, odds):
    if not odds or odds <= 1.0:
        return 0.0
    return (prob * odds - 1) * 100

DEFAULT_WEIGHTS = {
    "Real Madrid": {"attack": 1.35, "defense": 0.80},
    "Manchester City": {"attack": 1.40, "defense": 0.75},
    "FC Bayern München": {"attack": 1.30, "defense": 0.85},
    "FC Barcelona": {"attack": 1.25, "defense": 0.90},
    "Paris Saint-Germain": {"attack": 1.20, "defense": 0.90},
    "Inter": {"attack": 1.15, "defense": 0.80},
    "Arsenal": {"attack": 1.25, "defense": 0.80},
    "Borussia Dortmund": {"attack": 1.10, "defense": 1.00}
}

team_weights = load_json_file(WEIGHTS_FILE, DEFAULT_WEIGHTS)

# ==========================================
# NORMALIZACJA I REJESTRACJA NOWYCH DRUŻYN
# ==========================================
def normalize_team_name(name):
    mapping = {
        "Paris Saint Germain": "Paris Saint-Germain",
        "Bayern Munich": "FC Bayern München",
        "Barcelona": "FC Barcelona"
    }
    norm_name = mapping.get(name, name)
    
    # Jeśli drużyny nie ma w słowniku wag, dodaj ją automatycznie z domyślnymi wartościami
    if norm_name not in team_weights:
        team_weights[norm_name] = {"attack": 1.0, "defense": 1.0}
        save_json_file(WEIGHTS_FILE, team_weights)
        
    return norm_name

# ==========================================
# API: POBIERANIE KURSÓW ORAZ WYNIKÓW
# ==========================================
@st.cache_data(ttl=3600)
def fetch_ucl_odds(api_key):
    url = "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/odds/"
    params = {'apiKey': api_key, 'regions': 'eu', 'markets': 'h2h', 'dateFormat': 'iso'}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            matches = []
            for match in data:
                home_team = normalize_team_name(match.get('home_team'))
                away_team = normalize_team_name(match.get('away_team'))
                bookmakers = match.get('bookmakers', [])
                odds = {'1': None, 'X': None, '2': None}
                
                if bookmakers:
                    h2h_market = next((m for m in bookmakers[0]['markets'] if m['key'] == 'h2h'), None)
                    if h2h_market:
                        for outcome in h2h_market['outcomes']:
                            o_name = normalize_team_name(outcome['name'])
                            if o_name == home_team:
                                odds['1'] = outcome['price']
                            elif o_name == away_team:
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

def fetch_and_apply_automatic_results(api_key):
    url = "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/scores/"
    params = {'apiKey': api_key, 'daysFrom': 3}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            processed_ids = load_json_file(PROCESSED_MATCHES_FILE, [])
            updated = False

            for match in data:
                if match.get('completed', False) and match.get('id') not in processed_ids:
                    home_team = normalize_team_name(match.get('home_team'))
                    away_team = normalize_team_name(match.get('away_team'))
                    scores = match.get('scores', [])
                    
                    if scores and len(scores) == 2:
                        home_goals = int(next((s['score'] for s in scores if normalize_team_name(s['name']) == home_team), 0))
                        away_goals = int(next((s['score'] for s in scores if normalize_team_name(s['name']) == away_team), 0))

                        update_team_weights_residual_auto(home_team, away_team, home_goals, away_goals)
                        processed_ids.append(match.get('id'))
                        updated = True

            if updated:
                save_json_file(PROCESSED_MATCHES_FILE, processed_ids)
    except Exception:
        pass

def get_mock_matches():
    return [
        {
            "id": "mock_1",
            "home_team": "Real Madrid",
            "away_team": "Manchester City",
            "commence_time": "2026-11-26T20:00:00Z",
            "odds": {"1": 2.40, "X": 3.50, "2": 2.80}
        }
    ]

# ==========================================
# MODEL POISSONA I UCZENIE RESZTKOWE
# ==========================================
def predict_match(home_team, away_team, home_advantage=1.15, league_avg_goals=1.5):
    home_w = team_weights.get(home_team, {"attack": 1.0, "defense": 1.0})
    away_w = team_weights.get(away_team, {"attack": 1.0, "defense": 1.0})

    exp_home_goals = max(0.2, league_avg_goals * home_w["attack"] * away_w["defense"] * home_advantage)
    exp_away_goals = max(0.2, league_avg_goals * away_w["attack"] * home_w["defense"])

    max_goals = 8
    match_matrix = np.zeros((max_goals, max_goals))

    for h in range(max_goals):
        for a in range(max_goals):
            match_matrix[h, a] = poisson.pmf(h, exp_home_goals) * poisson.pmf(a, exp_away_goals)

    prob_home = np.sum(np.tril(match_matrix, -1))
    prob_draw = np.sum(np.diag(match_matrix))
    prob_away = np.sum(np.triu(match_matrix, 1))

    total_prob = prob_home + prob_draw + prob_away
    if total_prob > 0:
        prob_home /= total_prob
        prob_draw /= total_prob
        prob_away /= total_prob

    return {
        "exp_home_goals": round(exp_home_goals, 2),
        "exp_away_goals": round(exp_away_goals, 2),
        "prob_home": prob_home,
        "prob_draw": prob_draw,
        "prob_away": prob_away,
        "fair_odds_home": round(1 / prob_home, 2) if prob_home > 0 else 99,
        "fair_odds_draw": round(1 / prob_draw, 2) if prob_draw > 0 else 99,
        "fair_odds_away": round(1 / prob_away, 2) if prob_away > 0 else 99,
    }

def update_team_weights_residual_auto(home_team, away_team, actual_home_goals, actual_away_goals, learning_rate=0.05):
    pred = predict_match(home_team, away_team)
    
    err_home = actual_home_goals - pred["exp_home_goals"]
    err_away = actual_away_goals - pred["exp_away_goals"]

    if home_team not in team_weights:
        team_weights[home_team] = {"attack": 1.0, "defense": 1.0}
    if away_team not in team_weights:
        team_weights[away_team] = {"attack": 1.0, "defense": 1.0}

    team_weights[home_team]["attack"] = round(np.clip(team_weights[home_team]["attack"] + learning_rate * err_home, 0.5, 2.5), 3)
    team_weights[away_team]["defense"] = round(np.clip(team_weights[away_team]["defense"] + learning_rate * err_home, 0.5, 2.5), 3)

    team_weights[away_team]["attack"] = round(np.clip(team_weights[away_team]["attack"] + learning_rate * err_away, 0.5, 2.5), 3)
    team_weights[home_team]["defense"] = round(np.clip(team_weights[home_team]["defense"] + learning_rate * err_away, 0.5, 2.5), 3)

    save_json_file(WEIGHTS_FILE, team_weights)

fetch_and_apply_automatic_results(ODDS_API_KEY)

# ==========================================
# INTERFEJS UŻYTKOWNIKA STREAMLIT
# ==========================================
st.title("⚽ UCL Match Predictor & Value Betting Hub")
tab1, tab2, tab3 = st.tabs(["📅 Terminarz & Typy", "📊 Tabela & Statystyki Wag", "📝 Rejestr Zakładów & ROI"])

with tab1:
    st.header("Najbliższe Mecze i Wycena Kursów")
    matches = fetch_ucl_odds(ODDS_API_KEY)

    if not matches:
        st.info("Obecnie brak nadchodzących meczów Ligi Mistrzów w ofercie bukmacherów.")
    else:
        for match in matches:
            with st.expander(f"⚔️ {match['home_team']} vs {match['away_team']} — {match['commence_time'][:10]}", expanded=True):
                pred = predict_match(match['home_team'], match['away_team'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Prognoza goli (xG):**")
                    st.write(f"🏠 {match['home_team']}: **{pred['exp_home_goals']}**")
                    st.write(f"🚀 {match['away_team']}: **{pred['exp_away_goals']}**")

                with col2:
                    st.markdown("**Prawdopodobieństwo Modelu:**")
                    st.write(f"1: **{pred['prob_home']*100:.1f}%** (Fair: **{pred['fair_odds_home']}**)")
                    st.write(f"X: **{pred['prob_draw']*100:.1f}%** (Fair: **{pred['fair_odds_draw']}**)")
                    st.write(f"2: **{pred['prob_away']*100:.1f}%** (Fair: **{pred['fair_odds_away']}**)")

                with col3:
                    st.markdown("**Wartość (EV):**")
                    o_h, o_d, o_a = match['odds'].get('1'), match['odds'].get('X'), match['odds'].get('2')
                    ev_h, ev_d, ev_a = calculate_ev(pred['prob_home'], o_h), calculate_ev(pred['prob_draw'], o_d), calculate_ev(pred['prob_away'], o_a)
                    st.write(f"1: {o_h or '-'} | EV: **{ev_h:+.1f}%** {'🔥' if ev_h > 5 else ''}")
                    st.write(f"X: {o_d or '-'} | EV: **{ev_d:+.1f}%** {'🔥' if ev_d > 5 else ''}")
                    st.write(f"2: {o_a or '-'} | EV: **{ev_a:+.1f}%** {'🔥' if ev_a > 5 else ''}")

                st.divider()
                c1, c2, c3, _ = st.columns([2, 2, 2, 2])
                pick = c1.selectbox("Typuj:", ["1", "X", "2"], key=f"pick_{match['id']}")
                stake = c2.number_input("Stawka (PLN):", min_value=10, max_value=1000, value=100, key=f"stake_{match['id']}")
                odds_val = o_h if pick == "1" else (o_d if pick == "X" else o_a)
                
                if c3.button("➕ Dodaj do Rejestru", key=f"add_{match['id']}"):
                    history = load_json_file(BETS_FILE, [])
                    history.append({
                        "match": f"{match['home_team']} vs {match['away_team']}",
                        "pick": pick, "odds": odds_val or 2.0, "stake": stake,
                        "status": "PENDING", "profit": 0.0
                    })
                    save_json_file(BETS_FILE, history)
                    st.success("Dodano zakład!")

with tab2:
    st.header("Aktualna Siła Drużyn (Wagi)")
    st.info("ℹ️ Wagi ataku i obrony aktualizują się automatycznie w tle. Nowe drużyny z terminarza dopisują się do tabeli samoczynnie.")
    
    df_w = pd.DataFrame.from_dict(team_weights, orient='index').reset_index()
    df_w.columns = ["Drużyna", "Współczynnik Ataku", "Współczynnik Obrony (niżej = lepiej)"]
    st.dataframe(df_w, use_container_width=True)

with tab3:
    st.header("Rejestr Zakładów i ROI")
    bets = load_json_file(BETS_FILE, [])
    if not bets:
        st.info("Brak zapisanych zakładów.")
    else:
        df_b = pd.DataFrame(bets)
        settled = df_b[df_b['status'] != "PENDING"]
        st.metric("ROI", f"{(settled['profit'].sum() / settled['stake'].sum() * 100) if not settled.empty and settled['stake'].sum() > 0 else 0.0:+.2f}%")
        st.dataframe(df_b, use_container_width=True)
                
