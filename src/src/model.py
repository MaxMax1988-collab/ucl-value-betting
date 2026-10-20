import numpy as np
from scipy.stats import poisson

def predict_match(home_team, away_team, team_weights, home_advantage=1.15, league_avg_goals=1.5):
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

def update_team_weights_residual(home_team, away_team, actual_home_goals, actual_away_goals, team_weights, learning_rate=0.05):
    pred = predict_match(home_team, away_team, team_weights)
    
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

    return team_weights, err_home, err_away
  
