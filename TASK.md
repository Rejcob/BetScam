# Tarea: Implementar modelo Dixon-Coles para predicción de goles Over/Under 2.5

## Contexto del proyecto

Proyecto de modelado estadístico de fútbol (Premier League) para uso personal, sin fines
comerciales. Pipeline actual, ya funcionando:

- `load_data.py`: carga CSVs de football-data.co.uk a Postgres (tablas `teams`, `matches`, `odds`)
- `poisson_model.py`: modelo Poisson NAIVE (promedios simples de ataque/defensa por equipo).
  Ya identificamos que este modelo tiene sesgo sistemático y mala calibración (ver sección
  "Resultados del modelo naive" abajo) — el reemplazo por Dixon-Coles es para corregir esto.
- `evaluate.py`: calcula Brier Score y Log Loss comparando el modelo contra Pinnacle (benchmark)
  y contra dos baselines bobos (50/50 fijo, tasa base fija)
- `diagnose.py`: distribución de predicciones, calibración por bins, peores casos

## Schema de Postgres relevante

```sql
CREATE TABLE teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    league VARCHAR(10) NOT NULL,
    season VARCHAR(9) NOT NULL,
    match_date DATE NOT NULL,
    match_time TIME,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    home_goals_ft INTEGER NOT NULL,
    away_goals_ft INTEGER NOT NULL,
    result VARCHAR(1) NOT NULL,
    -- otras columnas: home_goals_ht, away_goals_ht, shots, corners, referee (no relevantes acá)
    UNIQUE(league, season, match_date, home_team_id, away_team_id)
);

CREATE TABLE odds (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    bookmaker VARCHAR(20) NOT NULL,   -- 'Pinnacle', 'Bet365'
    market VARCHAR(20) NOT NULL,      -- '1X2', 'OU2.5'
    timing VARCHAR(10) NOT NULL,      -- 'open', 'close'
    outcome VARCHAR(10) NOT NULL,     -- 'H'/'D'/'A' o 'Over'/'Under'
    odds_value DECIMAL(6,3) NOT NULL,
    UNIQUE(match_id, bookmaker, market, timing, outcome)
);
```

Conexión vía `.env` con variables `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
(usar `python-dotenv`, ya es el patrón del resto del proyecto).

## El problema con el modelo naive actual

El modelo actual calcula "fuerza de ataque/defensa" de cada equipo como un promedio simple de
goles marcados/recibidos, SIN considerar la calidad de los rivales enfrentados. Esto genera
sesgo: un equipo que jugó contra rivales débiles se ve artificialmente fuerte en ataque, y
viceversa en defensa.

**Resultados del modelo naive (para referencia, temporada 2024-2025, ventana "todos los
partidos anteriores"):**
- Brier Score del modelo: 0.2553
- Brier Score de Pinnacle (benchmark): 0.2435
- Brier Score de baseline bobo (tasa base fija): 0.2461
- **El modelo naive es PEOR que el baseline bobo** — no está capturando señal real.
- Calibración: en el rango de predicciones 70-100% (101 de 380 partidos), el modelo predice en
  promedio 74.8% pero la tasa real observada fue 60.4% — sobreconfianza sistemática.

## Qué hay que construir: `dixon_coles_model.py`

Reemplazar el cálculo naive de ataque/defensa por una estimación conjunta vía
**maximum likelihood estimation (MLE)**, siguiendo el enfoque de Dixon & Coles (1997).

### Fórmula del modelo (mantener esta estructura, ya validada conceptualmente)

Para un partido con equipo local `i` y visitante `j`:

```
goles_esperados_local  = mu * attack[i] * defense[j] * home_advantage
goles_esperados_visita = mu * attack[j] * defense[i]
```

Donde:
- `mu` = promedio de goles por equipo por partido en el histórico usado (NO se fitea, se
  calcula como promedio simple, igual que en el modelo naive)
- `attack[team]`, `defense[team]` = parámetros a estimar por MLE (relativos al promedio de
  liga, 1.0 = promedio)
- `home_advantage` = parámetro escalar único a estimar por MLE (aplica a todos los equipos
  por igual)

### Estimación (MLE)

1. Optimizar en dominio logarítmico para garantizar positividad: `attack[i] = exp(log_attack[i])`,
   ídem `defense[i]` y `home_advantage`.
2. **Restricción de identificabilidad**: fijar `attack` del primer equipo (orden alfabético) en
   1.0 (`log_attack = 0`), ya que el modelo tiene una redundancia de escala (multiplicar todos
   los `attack` por una constante y dividir todos los `defense` por la misma constante no
   cambia las predicciones).
3. Función objetivo: minimizar la log-verosimilitud negativa de Poisson sobre todos los partidos
   del historial:
   ```
   NLL = -sum( goles_local * log(lambda_local) - lambda_local - log(goles_local!)
             + goles_visita * log(lambda_visita) - lambda_visita - log(goles_visita!) )
   ```
   Usar `scipy.special.gammaln` para el término factorial (log-gamma, más estable que factorial
   directo).
4. **Regularización L2** sobre todos los parámetros (`log_attack`, `log_defense`,
   `log_home_advantage`), con coeficiente configurable (default sugerido: 0.1) — evita valores
   extremos cuando un equipo tiene poco historial (ej. inicio de temporada, equipos recién
   ascendidos).
5. Usar `scipy.optimize.minimize` con método `L-BFGS-B`.
6. **Vectorizar el cálculo de la NLL con numpy** (mapear equipos a índices enteros, arrays de
   goles), NO iterar partido por partido en un loop de Python — la velocidad importa porque el
   fit se corre una vez por cada fecha distinta de partidos en el set de predicción (ver abajo).

### Validación esperada (ya la corrimos en un caso sintético controlado, debe reproducirse)

Con datos sintéticos donde se conocen los valores reales de `attack`/`defense`/`home_advantage`
usados para generar los partidos, el fit debe recuperar valores razonablemente cercanos a los
reales (no exactos, hay ruido de muestra, pero la dirección y magnitud relativa entre equipos
debe ser consistente). Ejemplo de referencia que ya validamos manualmente:

```
Equipo       Attack estimado   Attack real   Defense estimado   Defense real
Man City     1.33              2.00          0.56                0.50
Burnley      0.37              0.50          2.05                1.60
```

(el orden de fuerza relativa entre equipos se preserva correctamente, aunque la magnitud
absoluta puede subestimarse un poco por la regularización — esto es aceptable y esperado)

### Reglas de backtesting — NO NEGOCIABLES

1. **Data leakage cero**: para predecir un partido de fecha `D`, el fit de Dixon-Coles debe usar
   ÚNICAMENTE partidos con `match_date < D`. Nunca usar partidos futuros ni el resultado del
   propio partido que se está prediciendo.
2. **Optimización de performance**: como todos los partidos de una misma fecha comparten el
   mismo corte de historial, agrupar por `match_date` y ajustar el modelo UNA VEZ por fecha
   distinta, no una vez por partido. Esto reduce el número de fits de ~380 a ~aprox. 35-40
   (una por jornada/fecha).
3. **Nunca tocar el set de test para ajustar hiperparámetros del modelo** (ej. el coeficiente de
   regularización L2) — si se quiere probar distintos valores, hacerlo con un split adicional de
   validación, no contra el CSV de predicciones que ya se usa para reportar métricas finales.

### CLI esperado (mismo patrón que `poisson_model.py`)

```bash
python dixon_coles_model.py --season 2024-2025 --league E0 --l2-reg 0.1
```

Debe generar un CSV `predictions_dixoncoles_<season>.csv` con las mismas columnas que
`poisson_model.py` ya genera (`match_id, date, home_team, away_team, actual_total_goals,
model_prob_over, model_prob_under, exp_home_goals, exp_away_goals`) más las columnas de odds de
Pinnacle desinfladas (`pinnacle_fair_prob_over`, `pinnacle_fair_prob_under`) — reutilizar
`attach_pinnacle_odds()` y `remove_overround()` de `poisson_model.py` tal cual están, sin
modificarlas (son correctas y ya están probadas).

### Criterio de éxito de esta tarea

Correr `python evaluate.py predictions_dixoncoles_2024-2025.csv` al final y confirmar:

1. El Brier Score del nuevo modelo es MENOR al del baseline bobo de tasa base (0.2461) — esto es
   el mínimo indispensable, si no se cumple el modelo sigue sin capturar señal real.
2. Reportar también la distancia al Brier Score de Pinnacle (0.2435) — no se espera ganarle en
   esta iteración, pero si la distancia se redujo respecto al modelo naive (que perdía por
   0.0117), es la señal de que el cambio estructural fue en la dirección correcta.
3. Correr `python diagnose.py predictions_dixoncoles_2024-2025.csv` y confirmar que la
   calibración en el rango 70-100% mejoró respecto al modelo naive (gap de 14 puntos entre
   predicho y real) — no hace falta que sea perfecta, pero debe reducirse.

## Fuera de alcance para esta tarea (explícitamente)

- NO conectar con ninguna casa de apuestas real ni automatizar registro de apuestas.
- NO implementar el ajuste "tau" de correlación de Dixon-Coles para resultados de bajo puntaje
  (0-0, 1-0, 0-1, 1-1) — es un refinamiento válido pero para una iteración futura.
- NO implementar decaimiento temporal (pesar partidos recientes más que viejos) — también
  queda para una iteración futura, mencionarlo como TODO en el código si es útil.
- NO iterar automáticamente probando distintos valores de `l2_reg` u otros hiperparámetros
  contra las métricas de `evaluate.py` buscando "ganar" — esa decisión se toma manualmente fuera
  de este ticket.
