import streamlit as st
import pandas as pd
from src.utils import load_json_file, save_json_file, calculate_ev
from src.api import fetch_ucl_odds
from src.model import predict_match, update_team_weights_residual

st.set_page_config(page_title="UCL Predictor & Value Betting Hub", page_icon="⚽", layout="wide")

ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", "TWOJ_KLUCZ_API")
WEIGHTS_FILE = "data/team_weights.json"
BETS_FILE = "data/bets_history.json"

DEFAULT_WEIGHTS = {
    "Real Madrid": {"attack": 1.35, "defense": 0.80},
    "Manchester City": {"attack": 1.40, "defense": 0.75},
    "FC Bayern München": {"attack": 1.30, "defense": 0.85},
    "FC Barcelona": {"attack": 1.25, "defense": 0.90}
}

team_weights = load_json_file(WEIGHTS_FILE, DEFAULT_WEIGHTS)

st.title("⚽ UCL Match Predictor & Value Betting Engine")
tab1, tab2, tab3 = st.tabs(["📅 Terminarz & Typy", "📊 Tabela & Statystyki", "📝 Rejestr & ROI"])

with tab1:
    st.header("Najbliższe Mecze i Wycena Kursów")
    matches = fetch_ucl_odds(ODDS_API_KEY)

    for match in matches:
        with st.expander(f"⚔️ {match['home_team']} vs {match['away_team']} — {match['commence_time'][:10]}", expanded=True):
            pred = predict_match(match['home_team'], match['away_team'], team_weights)
            
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
    st.header("Siła Drużyn (Wagi)")
    df_w = pd.DataFrame.from_dict(team_weights, orient='index').reset_index()
    df_w.columns = ["Drużyna", "Atak", "Obrona"]
    st.dataframe(df_w, use_container_width=True)

    st.subheader("⚙️ Symulator Uczenia Resztkowego")
    col_h, col_a, col_gh, col_ga = st.columns(4)
    sim_h = col_h.selectbox("Gospodarz", list(team_weights.keys()), key="sh")
    sim_a = col_a.selectbox("Gość", list(team_weights.keys()), key="sa")
    r_gh = col_gh.number_input("Gole Dom", 0, 10, 2, key="rgh")
    r_ga = col_ga.number_input("Gole Wyjazd", 0, 10, 1, key="rga")

    if st.button("🔄 Aktualizuj Wagi"):
        updated_w, eh, ea = update_team_weights_residual(sim_h, sim_a, r_gh, r_ga, team_weights)
        save_json_file(WEIGHTS_FILE, updated_w)
        st.success(f"Zaktualizowano wagi! Błąd resztkowy: {eh:+.2f} / {ea:+.2f}")
        st.rerun()

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
          
