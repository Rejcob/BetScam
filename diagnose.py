"""
Diagnóstico del modelo: ¿dónde y por qué está fallando?
Uso: python diagnose.py predictions_2024-2025_windowall.csv
"""
import sys
import pandas as pd
import numpy as np


def diagnose(csv_path):
    df = pd.read_csv(csv_path)
    df["actual_over"] = (df["actual_total_goals"] > 2.5).astype(int)
    df["error"] = (df["model_prob_over"] - df["actual_over"]).abs()

    print("=== 1. Distribución de las predicciones del modelo ===")
    print(df["model_prob_over"].describe())
    print(f"\n¿Cuántas predicciones son 'extremas' (>90% o <10%)?: "
          f"{((df['model_prob_over'] > 0.9) | (df['model_prob_over'] < 0.1)).sum()} de {len(df)}")

    print("\n=== 2. Calibración: cuando el modelo dice X%, ¿realmente pasa X% de las veces? ===")
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    df["bin"] = pd.cut(df["model_prob_over"], bins=bins)
    calibration = df.groupby("bin", observed=True).agg(
        n=("actual_over", "size"),
        predicho_promedio=("model_prob_over", "mean"),
        real_promedio=("actual_over", "mean"),
    )
    print(calibration.to_string())
    print("\n(Si 'predicho_promedio' y 'real_promedio' están lejos entre sí, el modelo está mal calibrado en ese rango)")

    print("\n=== 3. Los 10 peores casos (mayor error) ===")
    worst = df.nlargest(10, "error")[
        ["date", "home_team", "away_team", "actual_total_goals", "model_prob_over", "actual_over"]
    ]
    print(worst.to_string(index=False))

    print("\n=== 4. ¿Los equipos recién ascendidos están arruinando el promedio? ===")
    sospechosos = ["Ipswich", "Leicester", "Southampton"]
    mask = df["home_team"].isin(sospechosos) | df["away_team"].isin(sospechosos)
    if mask.sum() > 0:
        print(f"Partidos con equipos ascendidos: {mask.sum()}")
        print(f"Brier score EN esos partidos: {((df[mask]['model_prob_over'] - df[mask]['actual_over'])**2).mean():.4f}")
        print(f"Brier score SIN esos partidos: {((df[~mask]['model_prob_over'] - df[~mask]['actual_over'])**2).mean():.4f}")
    else:
        print("No se encontraron esos equipos en el dataset con esos nombres exactos.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python diagnose.py <ruta_al_csv_de_predicciones>")
        sys.exit(1)
    diagnose(sys.argv[1])
