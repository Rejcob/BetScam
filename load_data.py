
"""

Carga CSVs de football-data.co.uk a Postgres.

Uso: python load_data.py data/E0_2023-2024.csv 2023-2024

"""

import sys

import os

import pandas as pd

from sqlalchemy import create_engine, text

from dotenv import load_dotenv



load_dotenv()



DB_URL = (

    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"

    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

)



BOOKMAKERS = {

    "Pinnacle": {"prefix_1x2": "PS", "prefix_ou": "P"},

    "Bet365": {"prefix_1x2": "B365", "prefix_ou": "B365"},

}





def get_or_create_team(conn, name):

    result = conn.execute(

        text("SELECT id FROM teams WHERE name = :name"), {"name": name}

    ).fetchone()

    if result:

        return result[0]

    result = conn.execute(

        text("INSERT INTO teams (name) VALUES (:name) RETURNING id"),

        {"name": name},

    ).fetchone()

    return result[0]





def insert_odds(conn, match_id, bookmaker, market, timing, outcome, value):

    conn.execute(

        text("""

            INSERT INTO odds (match_id, bookmaker, market, timing, outcome, odds_value)

            VALUES (:match_id, :bookmaker, :market, :timing, :outcome, :value)

            ON CONFLICT (match_id, bookmaker, market, timing, outcome) DO NOTHING

        """),

        {

            "match_id": match_id,

            "bookmaker": bookmaker,

            "market": market,

            "timing": timing,

            "outcome": outcome,

            "value": float(value),

        },

    )

    return 1





def load_csv(csv_path, season, league="E0"):

    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])



    engine = create_engine(DB_URL)



    with engine.begin() as conn:

        matches_loaded = 0

        odds_loaded = 0



        for _, row in df.iterrows():

            home_id = get_or_create_team(conn, row["HomeTeam"])

            away_id = get_or_create_team(conn, row["AwayTeam"])



            match_date = pd.to_datetime(row["Date"], dayfirst=True).date()



            existing = conn.execute(

                text(

                    "SELECT id FROM matches WHERE league=:league AND season=:season "

                    "AND match_date=:date AND home_team_id=:home AND away_team_id=:away"

                ),

                {

                    "league": league,

                    "season": season,

                    "date": match_date,

                    "home": home_id,

                    "away": away_id,

                },

            ).fetchone()



            if existing:

                match_id = existing[0]

            else:

                result = conn.execute(

                    text("""

                        INSERT INTO matches (

                            league, season, match_date, match_time,

                            home_team_id, away_team_id,

                            home_goals_ft, away_goals_ft, result,

                            home_goals_ht, away_goals_ht,

                            home_shots, away_shots,

                            home_shots_target, away_shots_target,

                            home_corners, away_corners, referee

                        ) VALUES (

                            :league, :season, :date, :time,

                            :home, :away,

                            :fthg, :ftag, :ftr,

                            :hthg, :htag,

                            :hs, :as_,

                            :hst, :ast,

                            :hc, :ac, :ref

                        ) RETURNING id

                    """),

                    {

                        "league": league,

                        "season": season,

                        "date": match_date,

                        "time": row.get("Time"),

                        "home": home_id,

                        "away": away_id,

                        "fthg": int(row["FTHG"]),

                        "ftag": int(row["FTAG"]),

                        "ftr": row["FTR"],

                        "hthg": row.get("HTHG"),

                        "htag": row.get("HTAG"),

                        "hs": row.get("HS"),

                        "as_": row.get("AS"),

                        "hst": row.get("HST"),

                        "ast": row.get("AST"),

                        "hc": row.get("HC"),

                        "ac": row.get("AC"),

                        "ref": row.get("Referee"),

                    },

                ).fetchone()

                match_id = result[0]

                matches_loaded += 1



            for book_name, cfg in BOOKMAKERS.items():

                prefix_1x2 = cfg["prefix_1x2"]

                prefix_ou = cfg["prefix_ou"]



                for outcome, suffix in [("H", "H"), ("D", "D"), ("A", "A")]:

                    col = f"{prefix_1x2}{suffix}"

                    if col in row and pd.notna(row[col]):

                        odds_loaded += insert_odds(

                            conn, match_id, book_name, "1X2", "open", outcome, row[col]

                        )



                for outcome, suffix in [("H", "CH"), ("D", "CD"), ("A", "CA")]:

                    col = f"{prefix_1x2}{suffix}"

                    if col in row and pd.notna(row[col]):

                        odds_loaded += insert_odds(

                            conn, match_id, book_name, "1X2", "close", outcome, row[col]

                        )



                for outcome, suffix in [("Over", ">2.5"), ("Under", "<2.5")]:

                    col = f"{prefix_ou}{suffix}"

                    if col in row and pd.notna(row[col]):

                        odds_loaded += insert_odds(

                            conn, match_id, book_name, "OU2.5", "open", outcome, row[col]

                        )



                for outcome, suffix in [("Over", "C>2.5"), ("Under", "C<2.5")]:

                    col = f"{prefix_ou}{suffix}"

                    if col in row and pd.notna(row[col]):

                        odds_loaded += insert_odds(

                            conn, match_id, book_name, "OU2.5", "close", outcome, row[col]

                        )



        print(f"{csv_path}: {matches_loaded} partidos nuevos, {odds_loaded} odds cargadas.")





if __name__ == "__main__":

    if len(sys.argv) < 3:

        print("Uso: python load_data.py <ruta_csv> <temporada>")

        print("Ejemplo: python load_data.py data/E0_2023-2024.csv 2023-2024")

        sys.exit(1)



    load_csv(sys.argv[1], sys.argv[2])

