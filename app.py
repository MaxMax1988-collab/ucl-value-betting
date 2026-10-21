import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import re
from datetime import datetime, timezone
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
# BAZOWE WAGI NA PODSTAWIE UEFA & CLUB ELO
# ==========================================
DEFAULT_WEIGHTS = {
    "Real Madrid": {"attack": 1.35, "defense": 0.80},
    "Manchester City": {"attack": 1.40, "defense": 0.75},
    "FC Bayern München": {"attack": 1.30, "defense": 0.85},
    "FC Barcelona": {"attack": 1.25, "defense": 0.90},
    "Paris Saint-Germain": {"attack": 1.20, "defense": 0.90},
    "Inter": {"attack": 1.15, "defense": 0.80},
    "Arsenal": {"attack": 1.25, "defense": 0.80},
    "Liverpool": {"attack": 1.35, "defense": 0.85},
    "Borussia Dortmund": {"attack": 1.10, "defense": 1.00},
    "Napoli": {"attack": 1.15, "defense": 0.90},
    "Villarreal": {"attack": 1.05, "defense": 1.00},
    "FC Porto": {"attack": 1.05, "defense": 0.95},
    "PSV Eindhoven": {"attack": 1.10, "defense": 1.05},
    "Slavia Praha": {"attack": 0.95, "defense": 1.05},
    "Fenerbahce": {"attack": 1.05, "defense": 1.10},
    "Bodø/Glimt": {"attack": 0.90, "defense": 1.15},
    "Sabah FK": {"attack": 0.75, "defense": 1.30},
    "Leverkusen": {"attack": 1.20, "defense": 0.85},
    "Atalanta": {"attack": 1.15, "defense": 1.00},
    "AC Milan": {"attack": 1.10, "defense": 0.95},
    "Atletico Madrid": {"attack": 1.05, "defense": 0.80},
    "Juventus": {"attack": 1.00, "defense": 0.85},
    "Benfica": {"attack": 1.05, "defense": 0.95},
    "Sporting CP": {"attack": 1.10, "defense": 0.95},
    "Club Brugge": {"attack": 0.95, "defense": 1.10},
    "Celtic": {"attack": 0.90, "defense": 1.15},
    "Feyenoord": {"attack": 1.00, "defense": 1.05},
    "AS Roma": {"attack": 1.10, "defense": 0.85},
    "ŠK Slovan Bratislava": {"attack": 0.85, "defense": 1.20},
    "AEK Athens": {"attack": 0.95, "defense": 1.10},
    "VfB Stuttgart": {"attack": 1.10, "defense": 1.00},
    "Como": {"attack": 0.90, "defense": 1.10},
    "Manchester United": {"attack": 1.15, "defense": 0.95},
    "Lille": {"attack": 1.05, "defense": 0.95},
    "Galatasaray": {"attack": 1.10, "defense": 1.00},
    "Aston Villa": {"attack": 1.15, "defense": 0.90},
    "Viking FK": {"attack": 0.85, "defense": 1.15},
    "Inter Milan": {"attack": 1.15, "defense": 0.80},
    "Shakhtar Donetsk": {"attack": 1.00, "defense": 1.05},
    "Sporting Lisbon": {"attack": 1.10, "defense": 0.95},
    "LASK": {"attack": 0.90, "defense": 1.10}
}

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

def simplify_string(s):
    s = re.sub(r'\b(fc|fk|sk|ak)\b', '', s, flags=re.IGNORECASE)
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

team_weights = load_json_file(WEIGHTS_FILE, DEFAULT_WEIGHTS)
for t_name, t_vals in DEFAULT_WEIGHTS.items():
    if t_name not in team_weights:
        team_weights[t_name] = t_vals
save_json_file(WEIGHTS_FILE, team_weights)

def normalize_team_name(name):
    mapping = {
        "Paris Saint Germain": "Paris Saint-Germain",
        "Bayern Munich": "FC Bayern München",
        "Barcelona": "FC Barcelona",
        "Porto": "FC Porto",
        "Fenerbahçe": "Fenerbahce",
        "Slavia Prague": "Slavia Praha",
        "Bodo/Glimt": "Bodø/Glimt",
        "Bayer Leverkusen": "Leverkusen",
        "Atlético Madrid": "Atletico Madrid",
        "Slovan Bratislava": "ŠK Slovan Bratislava"
    }
    norm_name = mapping.get(name, name)
    if norm_name not in team_weights:
        team_weights[norm_name] = {"attack": 1.0, "defense": 1.0}
        save_json_file(WEIGHTS_FILE, team_weights)
    return norm_name

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

# ==========================================
# API: POBIERANJE KURSÓW ORAZ AUTONOMOIDALNE BOTOWANIE
# ==========================================
@st.cache_data(ttl=1800)
def fetch_ucl_odds_and_auto_bet(api_key):
    url = "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/odds/"
    params = {'apiKey': api_key, 'regions': 'eu', 'markets': 'h2h', 'dateFormat': 'iso'}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            matches = []
            bets = load_json_file(BETS_FILE, [])
            bets_changed = False

            for match in data:
                home_team = normalize_team_name(match.get('home_team'))
                away_team = normalize_team_name(match.get('away_team'))
                commence_time = match.get('commence_time')
                
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

                # Oblicz prognozę
                pred = predict_match(home_team, away_team)
                
                # Sprawdź Value (EV) dla każdego rynku w zdrowych granicach (EV: 5% do 25%, kursy: 1.80 do 3.50)
                market_evals = [
                    ("1", odds.get('1'), pred['prob_home']),
                    ("X", odds.get('X'), pred['prob_draw']),
                    ("2", odds.get('2'), pred['prob_away'])
                ]

                match_str = f"{home_team} vs {away_team}"

                for pick_type, odd_val, prob in market_evals:
                    if odd_val and 1.80 <= odd_val <= 3.50:
                        ev = calculate_ev(prob, odd_val)
                        if 5.0 <= ev <= 25.0: # Zdrowe value
                            # Sprawdź czy ten zakład już istnieje w rejestrze
                            exists = any(b.get("match") == match_str and b.get("pick") == pick_type for b in bets)
                            if not exists:
                                bets.append({
                                    "match": match_str,
                                    "pick": pick_type,
                                    "odds": odd_val,
                                    "stake": 100.0,
                                    "status": "PENDING",
                                    "profit": 0.0
                                })
                                bets_changed = True

                matches.append({
                    "id": match.get('id'),
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence_time,
                    "odds": odds,
                    "pred": pred
                })

            if bets_changed:
                save_json_file(BETS_FILE, bets)

            return matches
        else:
            return []
    except Exception:
        return []

def fetch_and_apply_automatic_results(api_key):
    url = "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/scores/"
    params = {'apiKey': api_key, 'daysFrom': 7}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            processed_ids = load_json_file(PROCESSED_MATCHES_FILE, [])
            bets = load_json_file(BETS_FILE, [])
            
            weights_updated = False
            bets_updated = False

            for match in data:
                if match.get('completed', False):
                    home_team = normalize_team_name(match.get('home_team'))
                    away_team = normalize_team_name(match.get('away_team'))
                    scores = match.get('scores', [])
                    
                    if scores and len(scores) == 2:
                        home_goals = int(next((s['score'] for s in scores if normalize_team_name(s['name']) == home_team), 0))
                        away_goals = int(next((s['score'] for s in scores if normalize_team_name(s['name']) == away_team), 0))

                        if match.get('id') not in processed_ids:
                            update_team_weights_residual_auto(home_team, away_team, home_goals, away_goals)
                            processed_ids.append(match.get('id'))
                            weights_updated = True

                        actual_result = "1" if home_goals > away_goals else ("2" if away_goals > home_goals else "X")
                        simp_home = simplify_string(home_team)
                        simp_away = simplify_string(away_team)

                        for bet in bets:
                            if bet.get("status") == "PENDING":
                                bet_match_simp = simplify_string(bet.get("match", ""))
                                if simp_home in bet_match_simp and simp_away in bet_match_simp:
                                    if bet.get("pick") == actual_result:
                                        bet["status"] = "WON"
                                        bet["profit"] = round(bet["stake"] * (bet["odds"] - 1), 2)
                                    else:
                                        bet["status"] = "LOST"
                                        bet["profit"] = -round(bet["stake"], 2)
                                    bets_updated = True

            if weights_updated:
                save_json_file(PROCESSED_MATCHES_FILE, processed_ids)
            if bets_updated:
                save_json_file(BETS_FILE, bets)
    except Exception:
        pass

# Uruchomienie automatycznego rozliczania wyników przy starcie
fetch_and_apply_automatic_results(ODDS_API_KEY)
matches = fetch_ucl_odds_and_auto_bet(ODDS_API_KEY)

# ==========================================
# INTERFEJS UŻYTKOWNIKA STREAMLIT
# ==========================================
st.title("⚽ Autonomiczny UCL Value Bot")
tab1, tab2, tab3 = st.tabs(["📅 Nadchodzące Mecze & AI", "📊 Tabela & Statystyki Wag", "📝 Autonomiczny Rejestr Zakładów & ROI"])

with tab1:
    st.header("Analiza i Automatyczne Wstrzykiwanie Value Betów")
    st.info("🤖 Bot samoczynnie skanuje kursy przedmeczowe, filtruje zdrowy EV (5%-25%) i bezpieczne kursy (1.80-3.50) do rejestru.")

    if not matches:
        st.warning("Brak aktywnych meczów Ligi Mistrzów lub problem z połączeniem z API.")
    else:
        for match in matches:
            with st.expander(f"⚔️ {match['home_team']} vs {match['away_team']} — {match['commence_time'][:10]}", expanded=True):
                pred = match['pred']
                o_h, o_d, o_a = match['odds'].get('1'), match['odds'].get('X'), match['odds'].get('2')
                
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
                    ev_h = calculate_ev(pred['prob_home'], o_h)
                    ev_d = calculate_ev(pred['prob_draw'], o_d)
                    ev_a = calculate_ev(pred['prob_away'], o_a)
                    
                    st.write(f"1: {o_h or '-'} | EV: **{ev_h:+.1f}%** {'🔥' if 5 <= ev_h <= 25 else ''}")
                    st.write(f"X: {o_d or '-'} | EV: **{ev_d:+.1f}%** {'🔥' if 5 <= ev_d <= 25 else ''}")
                    st.write(f"2: {o_a or '-'} | EV: **{ev_a:+.1f}%** {'🔥' if 5 <= ev_a <= 25 else ''}")

with tab2:
    st.header("Aktualna Siła Drużyn (Wagi)")
    st.info("ℹ️ Parametry ataku i obrony modyfikowane automatycznie przez uczenie resztkowe z wyników spotkań.")
    
    df_w = pd.DataFrame.from_dict(team_weights, orient='index').reset_index()
    df_w.columns = ["Drużyna", "Współczynnik Ataku", "Współczynnik Obrony (niżej = lepiej)"]
    st.dataframe(df_w, use_container_width=True)

with tab3:
    st.header("Rejestr Automatycznych Zakładów i ROI")
    bets = load_json_file(BETS_FILE, [])
    if not bets:
        st.info("Brak automatycznie dodanych zakładów spełniających kryteria zdrowego value.")
    else:
        df_b = pd.DataFrame(bets)
        settled = df_b[df_b['status'] != "PENDING"]
        
        total_stake = settled['stake'].sum() if not settled.empty else 0
        total_profit = settled['profit'].sum() if not settled.empty else 0
        roi_val = (total_profit / total_stake * 100) if total_stake > 0 else 0.0

        st.metric("ROI", f"{roi_val:+.2f}%", delta=f"{total_profit:+.2f} PLN")
        st.dataframe(df_b, use_container_width=True)
