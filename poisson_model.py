"""
Modelo Poisson simple para Over/Under 2.5 goles.
Uso: python poisson_model.py --window 10 --season 2024-2025
     python poisson_model.py --window all --season 2024-2025
"""
import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

MAX_GOALS = 15  # tope para la suma de la distribución conjunta


def load_all_matches(engine, league="E0"):
    """Trae todos los partidos ordenados por fecha, para calcular fuerzas 'a la fecha'."""
    query = text("""
        SELECT m.id, m.match_date, m.season,
               m.home_team_id, m.away_team_id,
               t1.name AS home_team, t2.name AS away_team,
               m.home_goals_ft, m.away_goals_ft
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.league = :league
        ORDER BY m.match_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"league": league})
    return df


def team_strengths(df_history, window):
    """
    Calcula fuerza de ataque/defensa de cada equipo, usando SOLO partidos
    en df_history (que el llamador ya filtró a 'antes de la fecha del partido').

    window=None -> usa todo df_history
    window=N    -> usa solo los últimos N partidos de cada equipo dentro de df_history
    """
    if len(df_history) == 0:
        return {}, 1.4

    avg_goals = (df_history["home_goals_ft"].mean() + df_history["away_goals_ft"].mean()) / 2

    teams = set(df_history["home_team"]) | set(df_history["away_team"])
    strengths = {}

    for team in teams:
        home_games = df_history[df_history["home_team"] == team]
        away_games = df_history[df_history["away_team"] == team]

        if window is not None:
            home_games = home_games.tail(window)
            away_games = away_games.tail(window)

        goals_for = pd.concat([home_games["home_goals_ft"], away_games["away_goals_ft"]])
        goals_against = pd.concat([home_games["away_goals_ft"], away_games["home_goals_ft"]])

        if len(goals_for) == 0:
            attack, defense = 1.0, 1.0
        else:
            attack = goals_for.mean() / avg_goals if avg_goals > 0 else 1.0
            defense = goals_against.mean() / avg_goals if avg_goals > 0 else 1.0

        strengths[team] = {"attack": attack, "defense": defense}

    return strengths, avg_goals


def predict_over_under(home_team, away_team, strengths, avg_goals, home_advantage=1.15):
    """Devuelve P(Over 2.5), P(Under 2.5) para un partido, dado el diccionario de fuerzas."""
    home_s = strengths.get(home_team, {"attack": 1.0, "defense": 1.0})
    away_s = strengths.get(away_team, {"attack": 1.0, "defense": 1.0})

    exp_home_goals = home_s["attack"] * away_s["defense"] * home_advantage * avg_goals
    exp_away_goals = away_s["attack"] * home_s["defense"] * avg_goals

    max_goals = max(MAX_GOALS, int((exp_home_goals + exp_away_goals) * 3))

    prob_over, prob_under = 0.0, 0.0
    for h in range(max_goals):
        for a in range(max_goals):
            p = poisson.pmf(h, exp_home_goals) * poisson.pmf(a, exp_away_goals)
            if h + a > 2.5:
                prob_over += p
            else:
                prob_under += p

    total = prob_over + prob_under
    if total < 0.999:
        print(f"Advertencia: suma de probabilidades = {total:.4f}, revisar goles esperados: {exp_home_goals:.2f}/{exp_away_goals:.2f}")

    return prob_over, prob_under, exp_home_goals, exp_away_goals


def remove_overround(odds_over, odds_under):
    """Convierte odds decimales con margen de casa a probabilidades 'verdaderas' (suman 1)."""
    implied_over = 1 / odds_over
    implied_under = 1 / odds_under
    total = implied_over + implied_under
    return implied_over / total, implied_under / total


def run_predictions(engine, window, target_season, league="E0"):
    df = load_all_matches(engine, league)
    df["match_date"] = pd.to_datetime(df["match_date"])

    target_matches = df[df["season"] == target_season].copy()

    results = []

    for _, match in target_matches.iterrows():
        history = df[df["match_date"] < match["match_date"]]

        strengths, avg_goals = team_strengths(history, window)
        prob_over, prob_under, exp_h, exp_a = predict_over_under(
            match["home_team"], match["away_team"], strengths, avg_goals
        )

        results.append({
            "match_id": match["id"],
            "date": match["match_date"].date(),
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "actual_total_goals": match["home_goals_ft"] + match["away_goals_ft"],
            "model_prob_over": round(prob_over, 4),
            "model_prob_under": round(prob_under, 4),
            "exp_home_goals": round(exp_h, 2),
            "exp_away_goals": round(exp_a, 2),
        })

    return pd.DataFrame(results)


def attach_pinnacle_odds(engine, predictions_df):
    """Suma las odds de cierre de Pinnacle a cada partido, ya sin overround."""
    match_ids = predictions_df["match_id"].tolist()
    query = text("""
        SELECT match_id, outcome, odds_value
        FROM odds
        WHERE bookmaker = 'Pinnacle' AND market = 'OU2.5' AND timing = 'close'
        AND match_id = ANY(:ids)
    """)
    with engine.connect() as conn:
        odds_df = pd.read_sql(query, conn, params={"ids": match_ids})

    pivot = odds_df.pivot(index="match_id", columns="outcome", values="odds_value")
    predictions_df = predictions_df.merge(pivot, left_on="match_id", right_index=True, how="left")

    fair_probs = predictions_df.apply(
        lambda r: remove_overround(r["Over"], r["Under"])
        if pd.notna(r.get("Over")) and pd.notna(r.get("Under"))
        else (np.nan, np.nan),
        axis=1,
        result_type="expand",
    )
    predictions_df["pinnacle_fair_prob_over"] = fair_probs[0]
    predictions_df["pinnacle_fair_prob_under"] = fair_probs[1]

    return predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="all", help="'all' o un número, ej. 10")
    parser.add_argument("--season", required=True, help="ej. 2024-2025")
    parser.add_argument("--league", default="E0")
    args = parser.parse_args()

    window = None if args.window == "all" else int(args.window)

    engine = create_engine(DB_URL)
    predictions = run_predictions(engine, window, args.season, args.league)
    predictions = attach_pinnacle_odds(engine, predictions)

    out_path = f"predictions_{args.season}_window{args.window}.csv"
    predictions.to_csv(out_path, index=False)

    print(f"\n{len(predictions)} predicciones generadas -> {out_path}")
    print(predictions[[
        "date", "home_team", "away_team", "actual_total_goals",
        "model_prob_over", "pinnacle_fair_prob_over"
    ]].head(10).to_string(index=False))
