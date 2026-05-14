"""
Detección de quiebres estructurales en series temporales de consumo de café.

Usa el algoritmo PELT (Pruned Exact Linear Time) sobre la pendiente de la
serie. Filtra quiebres espurios mediante reglas heurísticas:
- Longitud mínima de segmento: 5 años
- Cambio mínimo de pendiente: 30% relativo (filtra ruido)
- Solo se detecta UN quiebre principal (el más significativo)

Salida por país:
- breakpoint_year: año del quiebre, o None
- pre_slope: pendiente antes del quiebre
- post_slope: pendiente después del quiebre
- slope_change_pct: cambio relativo de pendiente
- significance: clasificación cualitativa (none / mild / strong / reversal)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import ruptures as rpt

MIN_SEGMENT = 5      # años mínimos por segmento
SLOPE_THRESHOLD = 30 # % mínimo de cambio para considerar quiebre real
PELT_PENALTY = 3.0   # penalty del PELT (calibrado para n=30)


def detect_one_break(years, values):
    """
    Detecta el quiebre estructural más significativo (si existe).

    Returns:
        dict con keys: breakpoint_year, pre_slope, post_slope,
        slope_change_pct, significance
    """
    # Si la serie tiene muchos ceros o constantes, no aplica
    if np.std(values) == 0 or len(values) < 2 * MIN_SEGMENT:
        return _no_break(years, values)

    # Detrend para que PELT capture cambios de PENDIENTE, no de nivel.
    # Usamos diferencias.
    diffs = np.diff(values).reshape(-1, 1)

    # PELT con modelo de cambio de media en las diferencias
    algo = rpt.Pelt(model="l2", min_size=MIN_SEGMENT, jump=1).fit(diffs)
    breakpoints = algo.predict(pen=PELT_PENALTY)
    # breakpoints incluye el último índice como cierre — lo quitamos
    candidates = [b for b in breakpoints[:-1] if MIN_SEGMENT <= b <= len(diffs) - MIN_SEGMENT]

    if not candidates:
        return _no_break(years, values)

    # Si hay múltiples quiebres, elegimos el que produce el mayor cambio
    # de pendiente entre segmentos contiguos
    best_break = None
    best_change = 0
    for b in candidates:
        # b es índice en diffs; el quiebre ocurre entre años[b] y años[b+1]
        pre = values[: b + 1]
        post = values[b + 1 :]
        if len(pre) < MIN_SEGMENT or len(post) < MIN_SEGMENT:
            continue
        slope_pre = _slope(years[: b + 1], pre)
        slope_post = _slope(years[b + 1 :], post)
        if abs(slope_pre) < 1:  # evitar división explosiva
            change = abs(slope_post - slope_pre)
        else:
            change = abs((slope_post - slope_pre) / slope_pre) * 100
        if change > best_change:
            best_change = change
            best_break = (b, slope_pre, slope_post)

    if best_break is None or best_change < SLOPE_THRESHOLD:
        return _no_break(years, values)

    b, slope_pre, slope_post = best_break
    break_year = years[b + 1]

    # Clasificar la significancia
    if (slope_pre > 0 and slope_post < 0) or (slope_pre < 0 and slope_post > 0):
        sig = "reversal"
    elif best_change > 100:
        sig = "strong"
    elif best_change > 50:
        sig = "moderate"
    else:
        sig = "mild"

    return {
        "breakpoint_year": int(break_year),
        "pre_slope": float(slope_pre),
        "post_slope": float(slope_post),
        "slope_change_pct": float(best_change),
        "significance": sig,
    }


def _slope(years, values):
    """Pendiente OLS simple."""
    if len(years) < 2:
        return 0
    return float(np.polyfit(years, values, 1)[0])


def _no_break(years, values):
    return {
        "breakpoint_year": None,
        "pre_slope": _slope(years, values),
        "post_slope": _slope(years, values),
        "slope_change_pct": 0.0,
        "significance": "none",
    }


def detect_all_breaks(long_df):
    """
    Aplica detección de quiebres a todas las series país-tipo.

    Returns:
        DataFrame con una fila por país-tipo y columnas del resultado.
    """
    results = []
    for (country, ctype), grp in long_df.groupby(["Country", "Coffee type"]):
        grp = grp.sort_values("year")
        years = grp["year"].values
        values = grp["consumption_cups"].values

        # Manejo de ceros: si hay ceros al inicio, recortar
        nonzero_mask = values > 0
        if not nonzero_mask.any():
            continue
        first_nonzero = np.argmax(nonzero_mask)
        years_eff = years[first_nonzero:]
        values_eff = values[first_nonzero:]

        if len(values_eff) < 2 * MIN_SEGMENT:
            res = _no_break(years_eff, values_eff)
        else:
            res = detect_one_break(years_eff, values_eff)

        res["Country"] = country
        res["Coffee type"] = ctype
        res["n_observations"] = len(values_eff)
        results.append(res)

    return pd.DataFrame(results)


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    if (BASE.parent / "data").exists() and BASE.name == "src":
        BASE = BASE.parent

    long_df = pd.read_parquet(BASE / "data" / "coffee_long.parquet")
    breaks = detect_all_breaks(long_df)

    print(f"Quiebres detectados: {(breaks['breakpoint_year'].notna()).sum()} / {len(breaks)} países")
    print("\nDistribución por significancia:")
    print(breaks["significance"].value_counts())

    print("\nTop 10 quiebres más fuertes:")
    print(breaks[breaks["breakpoint_year"].notna()]
          .nlargest(10, "slope_change_pct")
          [["Country", "breakpoint_year", "pre_slope", "post_slope",
            "slope_change_pct", "significance"]]
          .to_string())

    out = BASE / "data" / "breakpoints.parquet"
    breaks.to_parquet(out, index=False)
    print(f"\n✓ Guardado: {out}")