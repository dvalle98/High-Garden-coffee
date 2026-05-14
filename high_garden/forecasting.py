"""
Motor de forecasting para consumo doméstico de café.

Cuatro modelos compiten por país en backtest, gana el de menor MAPE:
    1. Naive con drift     — baseline obligatorio
    2. Linear trend         — regresión lineal sobre el tiempo
    3. Holt (suavizado)     — exponencial con tendencia (statsmodels)
    4. Segmented linear     — respeta quiebres estructurales detectados

Evaluación: walk-forward con holdout de 5 años (2015-2019).
Horizonte de proyección final: 5 años (2020/21 - 2024/25).
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HOLDOUT = 5            # años de validación
HORIZON = 5            # años de proyección final
MIN_HISTORY = 8        # años mínimos de historia para modelar
CI_LEVEL = 0.80        # nivel de confianza para intervalos


# ============================================================================
# MODELOS
# ============================================================================

def naive_drift(history, h, return_ci=False):
    """
    Forecast naive con drift: extrapola la pendiente promedio reciente.
    f(t+h) = y_n + h * (y_n - y_1) / (n - 1)
    """
    y = np.asarray(history, dtype=float)
    n = len(y)
    drift = (y[-1] - y[0]) / (n - 1) if n > 1 else 0
    forecast = np.array([y[-1] + (i + 1) * drift for i in range(h)])
    if return_ci:
        # CI: 1.28 σ para 80%
        sigma = np.std(np.diff(y))
        spread = 1.28 * sigma * np.sqrt(np.arange(1, h + 1))
        return forecast, forecast - spread, forecast + spread
    return forecast


def linear_trend(history, h, return_ci=False):
    """Regresión lineal simple sobre índice temporal."""
    y = np.asarray(history, dtype=float)
    n = len(y)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)
    forecast = intercept + slope * np.arange(n, n + h)
    if return_ci:
        residuals = y - (intercept + slope * x)
        sigma = np.std(residuals)
        spread = 1.28 * sigma * np.sqrt(1 + 1 / n)
        return forecast, forecast - spread, forecast + spread
    return forecast


def holt(history, h, return_ci=False):
    """Suavizado exponencial con tendencia (Holt's linear trend)."""
    y = np.asarray(history, dtype=float)
    if len(y) < 4:
        return naive_drift(history, h, return_ci)
    try:
        model = ExponentialSmoothing(y, trend="add", seasonal=None,
                                     initialization_method="estimated")
        fit = model.fit(optimized=True, use_brute=True)
        forecast = fit.forecast(h)
        if return_ci:
            residuals = y - fit.fittedvalues
            sigma = np.std(residuals)
            spread = 1.28 * sigma * np.sqrt(np.arange(1, h + 1))
            return forecast, forecast - spread, forecast + spread
        return forecast
    except Exception:
        return naive_drift(history, h, return_ci)


def segmented_linear(history, h, breakpoint_idx, return_ci=False):
    """
    Regresión lineal sobre el segmento POST-quiebre solamente.
    Si no hay quiebre o el post-segmento es muy corto, cae a linear_trend.
    """
    y = np.asarray(history, dtype=float)
    n = len(y)

    if breakpoint_idx is None or breakpoint_idx < 0 or breakpoint_idx > n - 4:
        return linear_trend(history, h, return_ci)

    # Usar solo el post-segmento
    post_y = y[breakpoint_idx:]
    post_n = len(post_y)
    if post_n < 4:
        return linear_trend(history, h, return_ci)

    x = np.arange(post_n)
    slope, intercept = np.polyfit(x, post_y, 1)
    forecast = intercept + slope * np.arange(post_n, post_n + h)
    if return_ci:
        residuals = post_y - (intercept + slope * x)
        sigma = np.std(residuals)
        spread = 1.28 * sigma * np.sqrt(1 + 1 / post_n)
        return forecast, forecast - spread, forecast + spread
    return forecast


# ============================================================================
# MÉTRICAS
# ============================================================================

def mape(actual, predicted):
    """Mean Absolute Percentage Error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


# ============================================================================
# BACKTEST
# ============================================================================

def backtest_country(years, values, breakpoint_year=None):
    """
    Compite los 4 modelos en una ventana de holdout.

    Returns:
        dict por modelo con MAPE y RMSE; mejor modelo señalado.
    """
    # Filtrar ceros iniciales
    values = np.asarray(values, dtype=float)
    nonzero = values > 0
    if not nonzero.any():
        return None
    first_nz = np.argmax(nonzero)
    years_eff = np.asarray(years[first_nz:])
    values_eff = values[first_nz:]

    if len(values_eff) < MIN_HISTORY + HOLDOUT:
        return None

    train = values_eff[:-HOLDOUT]
    test = values_eff[-HOLDOUT:]

    # Index del quiebre dentro del training set
    bp_idx = None
    if breakpoint_year is not None:
        train_years = years_eff[:-HOLDOUT]
        match = np.where(train_years == breakpoint_year)[0]
        if len(match) > 0:
            bp_idx = int(match[0])

    results = {}
    for name, fn in [("naive_drift", naive_drift),
                     ("linear_trend", linear_trend),
                     ("holt", holt)]:
        try:
            pred = fn(train, HOLDOUT)
            results[name] = {
                "mape": mape(test, pred),
                "rmse": rmse(test, pred),
                "predictions": pred.tolist(),
            }
        except Exception as e:
            results[name] = {"mape": np.nan, "rmse": np.nan,
                             "predictions": None, "error": str(e)}

    # Modelo segmentado: solo si hay quiebre detectado dentro del train
    if bp_idx is not None and bp_idx > 0 and bp_idx < len(train) - 4:
        try:
            pred = segmented_linear(train, HOLDOUT, bp_idx)
            results["segmented"] = {
                "mape": mape(test, pred),
                "rmse": rmse(test, pred),
                "predictions": pred.tolist(),
            }
        except Exception as e:
            results["segmented"] = {"mape": np.nan, "error": str(e)}

    # Mejor modelo (menor MAPE válido)
    valid = {k: v for k, v in results.items() if not np.isnan(v.get("mape", np.nan))}
    if valid:
        best = min(valid.items(), key=lambda kv: kv[1]["mape"])
        results["best_model"] = best[0]
        results["best_mape"] = best[1]["mape"]
    else:
        results["best_model"] = None
        results["best_mape"] = np.nan

    results["test_actual"] = test.tolist()
    results["train_last_year"] = int(years_eff[-HOLDOUT - 1])
    return results


def backtest_all(long_df, breaks_df):
    """Aplica backtest a todos los países × tipos."""
    breaks_map = breaks_df.set_index(["Country", "Coffee type"])["breakpoint_year"].to_dict()

    summary = []
    detail = {}

    for (country, ctype), grp in long_df.groupby(["Country", "Coffee type"]):
        grp = grp.sort_values("year")
        years = grp["year"].values
        values = grp["consumption_cups"].values
        bp_year = breaks_map.get((country, ctype))
        if pd.isna(bp_year):
            bp_year = None

        res = backtest_country(years, values, bp_year)
        if res is None:
            continue

        key = f"{country} | {ctype}"
        detail[key] = res

        row = {
            "Country": country,
            "Coffee type": ctype,
            "best_model": res["best_model"],
            "best_mape": res["best_mape"],
        }
        for m in ["naive_drift", "linear_trend", "holt", "segmented"]:
            if m in res:
                row[f"{m}_mape"] = res[m].get("mape", np.nan)
        summary.append(row)

    return pd.DataFrame(summary), detail


# ============================================================================
# FORECAST FINAL
# ============================================================================

def forecast_all(long_df, breaks_df, backtest_summary):
    """
    Genera forecasts para 2020/21 - 2024/25 usando el mejor modelo por país.
    """
    breaks_map = breaks_df.set_index(["Country", "Coffee type"])["breakpoint_year"].to_dict()
    best_map = backtest_summary.set_index(["Country", "Coffee type"])["best_model"].to_dict()

    forecasts = []

    for (country, ctype), grp in long_df.groupby(["Country", "Coffee type"]):
        grp = grp.sort_values("year")
        years = grp["year"].values
        values = grp["consumption_cups"].values.astype(float)

        # Filtrar ceros iniciales
        nonzero = values > 0
        if not nonzero.any():
            continue
        first_nz = np.argmax(nonzero)
        years_eff = years[first_nz:]
        values_eff = values[first_nz:]
        if len(values_eff) < MIN_HISTORY:
            continue

        last_year = int(years_eff[-1])
        bp_year = breaks_map.get((country, ctype))
        best = best_map.get((country, ctype), "naive_drift")

        # Index del quiebre en la serie completa
        bp_idx = None
        if bp_year is not None and not pd.isna(bp_year):
            match = np.where(years_eff == int(bp_year))[0]
            if len(match) > 0:
                bp_idx = int(match[0])

        try:
            if best == "naive_drift":
                point, low, high = naive_drift(values_eff, HORIZON, return_ci=True)
            elif best == "linear_trend":
                point, low, high = linear_trend(values_eff, HORIZON, return_ci=True)
            elif best == "holt":
                point, low, high = holt(values_eff, HORIZON, return_ci=True)
            elif best == "segmented":
                point, low, high = segmented_linear(values_eff, HORIZON, bp_idx, return_ci=True)
            else:
                point, low, high = naive_drift(values_eff, HORIZON, return_ci=True)
        except Exception:
            point, low, high = naive_drift(values_eff, HORIZON, return_ci=True)

        # SAFETY CONSTRAINT: el consumo de café es "sticky" — no colapsa a cero.
        # Si el forecast de un año cae más del 30% respecto al año previo del
        # histórico/forecast, lo limitamos a 30% de caída por año.
        # Esto evita extrapolaciones lineales catastróficas (ej. Venezuela).
        FLOOR_RATIO = 0.70  # mínimo 70% del valor previo
        prev = values_eff[-1]
        for i in range(HORIZON):
            floor = prev * FLOOR_RATIO
            if point[i] < floor:
                point[i] = floor
                # Recalcular CI proporcionalmente
                low[i] = max(low[i], floor * 0.9)
                high[i] = max(high[i], floor * 1.1)
            prev = point[i]

        # Forzar no-negativos
        point = np.maximum(point, 0)
        low = np.maximum(low, 0)
        high = np.maximum(high, 0)

        for i in range(HORIZON):
            forecasts.append({
                "Country": country,
                "Coffee type": ctype,
                "year": last_year + i + 1,
                "forecast": float(point[i]),
                "lower_80": float(low[i]),
                "upper_80": float(high[i]),
                "model_used": best,
                "had_breakpoint": bp_idx is not None,
            })

    return pd.DataFrame(forecasts)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    if (BASE.parent / "data").exists() and BASE.name == "src":
        BASE = BASE.parent

    long_df = pd.read_parquet(BASE / "data" / "coffee_long.parquet")
    breaks_df = pd.read_parquet(BASE / "data" / "breakpoints.parquet")

    print("=" * 70)
    print("MOTOR DE FORECASTING")
    print("=" * 70)

    print("\n[1/2] Ejecutando backtest (holdout 5 años, 4 modelos)...")
    bt_summary, bt_detail = backtest_all(long_df, breaks_df)
    print(f"  Países modelados: {len(bt_summary)}")
    print("\n  Distribución del modelo ganador:")
    print(bt_summary["best_model"].value_counts().to_string())
    print(f"\n  MAPE mediano global: {bt_summary['best_mape'].median():.2f}%")
    print(f"  MAPE promedio global: {bt_summary['best_mape'].mean():.2f}%")

    # Comparación crítica: ¿el segmentado mejora en países CON quiebre?
    countries_with_break = breaks_df[breaks_df["breakpoint_year"].notna()].set_index(["Country", "Coffee type"]).index
    bt_with_break = bt_summary.set_index(["Country", "Coffee type"]).loc[
        bt_summary.set_index(["Country", "Coffee type"]).index.intersection(countries_with_break)
    ]
    print(f"\n  Países con quiebre: {len(bt_with_break)}")
    if "segmented_mape" in bt_with_break.columns:
        comparable = bt_with_break.dropna(subset=["segmented_mape", "linear_trend_mape"])
        wins_seg = (comparable["segmented_mape"] < comparable["linear_trend_mape"]).sum()
        print(f"  Donde 'segmented' bate a 'linear_trend': {wins_seg} de {len(comparable)} "
              f"({wins_seg/len(comparable)*100:.0f}%)")

    print("\n[2/2] Generando forecasts 2020-2024...")
    forecasts = forecast_all(long_df, breaks_df, bt_summary)
    print(f"  Filas de forecast: {len(forecasts)}")
    print(f"  Países con forecast: {forecasts['Country'].nunique()}")

    # Guardar
    bt_summary.to_parquet(BASE / "data" / "backtest_summary.parquet", index=False)
    forecasts.to_parquet(BASE / "data" / "forecasts.parquet", index=False)
    print(f"\n✓ Guardados en {BASE / 'data'}")

    print("\nTop 10 países con mayor crecimiento proyectado (2024 vs 2019):")
    last_actual = long_df[long_df["year"] == 2019].set_index(["Country", "Coffee type"])["consumption_cups"]
    last_forecast = forecasts[forecasts["year"] == 2024].set_index(["Country", "Coffee type"])["forecast"]
    growth = ((last_forecast - last_actual) / last_actual * 100).sort_values(ascending=False).head(10)
    print(growth.to_string())