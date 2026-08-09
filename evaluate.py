"""
Evalúa las predicciones generadas por poisson_model.py contra resultados reales.
Uso: python evaluate.py predictions_2024-2025_windowall.csv
"""
import sys
import pandas as pd
import numpy as np


def brier_score(probs, actuals):
    """Más bajo = mejor. 0 = predicción perfecta. 0.25 = tan bueno como decir siempre 50%."""
    return np.mean((probs - actuals) ** 2)


def log_loss(probs, actuals, eps=1e-15):
    """Más bajo = mejor. Castiga fuerte estar muy confiado y equivocado."""
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(actuals * np.log(probs) + (1 - actuals) * np.log(1 - probs))


def evaluate(csv_path):
    df = pd.read_csv(csv_path)

    df["actual_over"] = (df["actual_total_goals"] > 2.5).astype(int)

    df_valid = df.dropna(subset=["pinnacle_fair_prob_over"]).copy()
    n_dropped = len(df) - len(df_valid)

    print(f"Partidos totales: {len(df)}")
    if n_dropped > 0:
        print(f"Partidos sin odds de Pinnacle (excluidos de la comparación): {n_dropped}")
    print(f"Partidos evaluados: {len(df_valid)}")
    print(f"Tasa real de Over 2.5 en la muestra: {df_valid['actual_over'].mean():.3f}\n")

    model_brier = brier_score(df_valid["model_prob_over"].values, df_valid["actual_over"].values)
    model_logloss = log_loss(df_valid["model_prob_over"].values, df_valid["actual_over"].values)

    pinnacle_brier = brier_score(df_valid["pinnacle_fair_prob_over"].values, df_valid["actual_over"].values)
    pinnacle_logloss = log_loss(df_valid["pinnacle_fair_prob_over"].values, df_valid["actual_over"].values)

    naive_probs = np.full(len(df_valid), 0.5)
    naive_brier = brier_score(naive_probs, df_valid["actual_over"].values)
    naive_logloss = log_loss(naive_probs, df_valid["actual_over"].values)

    base_rate = df_valid["actual_over"].mean()
    base_probs = np.full(len(df_valid), base_rate)
    base_brier = brier_score(base_probs, df_valid["actual_over"].values)
    base_logloss = log_loss(base_probs, df_valid["actual_over"].values)

    print(f"{'Método':<25} {'Brier Score':>12} {'Log Loss':>12}")
    print("-" * 51)
    print(f"{'Tu modelo (Poisson)':<25} {model_brier:>12.4f} {model_logloss:>12.4f}")
    print(f"{'Pinnacle (benchmark)':<25} {pinnacle_brier:>12.4f} {pinnacle_logloss:>12.4f}")
    print(f"{'Bobo (siempre 50%)':<25} {naive_brier:>12.4f} {naive_logloss:>12.4f}")
    print(f"{'Bobo (tasa base)':<25} {base_brier:>12.4f} {base_logloss:>12.4f}")

    print("\n--- Interpretación ---")
    if model_brier < base_brier:
        print("Tu modelo le gana al baseline de tasa base -- hay señal real, no es puro ruido.")
    else:
        print("Tu modelo NO le gana al baseline de tasa base -- probablemente no está capturando nada útil todavía.")

    if model_brier < pinnacle_brier:
        print("Tu modelo le gana a Pinnacle en esta muestra -- interesante, pero con pocos partidos hay que ser cauteloso, podría ser ruido de muestra chica.")
    else:
        diff = model_brier - pinnacle_brier
        print(f"Pinnacle te gana por {diff:.4f} en Brier score -- esperable en la primera versión, es tu vara para mejorar.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python evaluate.py <ruta_al_csv_de_predicciones>")
        sys.exit(1)
    evaluate(sys.argv[1])
