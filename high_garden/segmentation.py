"""
Motor de segmentación estratégica y scoring de presión exportable.

Pipeline:
    1. Feature engineering por país (agregando tipos de café)
    2. Clustering K-Means con selección de K por silhouette
    3. Naming de clústers basado en perfiles
    4. Cálculo del Índice de Presión sobre Oferta (IPS)
    5. Ranking final accionable

Features por país (8 dimensiones):
    - size_log         : log(mean_consumption)
    - growth_forecast  : CAGR proyectado 2019 → 2024
    - growth_historical: CAGR 1990 → 2019
    - acceleration     : growth_forecast - growth_historical
    - cv               : coeficiente de variación
    - predictability   : 100 - best_mape (capped 0-100)
    - had_breakpoint   : 0/1
    - share_global     : % del consumo mundial actual
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================================
# PESOS DEL ÍNDICE DE PRESIÓN SOBRE OFERTA (IPS)
# ----------------------------------------------------------------------------
# Justificación de cada peso:
#   size_log         (30%): el tamaño absoluto importa más que cualquier %
#   growth_forecast  (30%): el componente forward-looking es central
#   acceleration     (20%): aceleración reciente = señal temprana de tensión
#   predictability   (20%): si no podemos confiar en el forecast, baja la urgencia
# ============================================================================
IPS_WEIGHTS = {
    "size_log":         0.30,
    "growth_forecast":  0.30,
    "acceleration":     0.20,
    "predictability":   0.20,
}

# Número de clústers a evaluar.
# IMPORTANTE: empezamos en K=4 (no en K=2 o K=3) por decisión deliberada.
# Brasil es un outlier extremo (8x más grande que el siguiente país). Con K<=3,
# el silhouette score "premia" aislar a Brasil en su propio clúster, dejando
# todos los demás en un grupo grande. Eso destruye la utilidad de negocio.
# Empezamos en K=4 para forzar al algoritmo a producir segmentación útil.
K_RANGE = range(4, 8)
RANDOM_STATE = 42


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_country_features(long_df, wide_df, breaks_df, bt_summary, forecasts):
    """
    Construye matriz de features a nivel país (agregando coffee types).
    Returns: DataFrame indexado por Country con 8 features + metadata.
    """
    # 1. Agregar a nivel país (sumando tipos)
    country_yearly = (long_df.groupby(["Country", "year"])["consumption_cups"]
                      .sum().reset_index())

    # 2. Métricas básicas
    country_stats = (country_yearly.groupby("Country")["consumption_cups"]
                     .agg(["mean", "std", "min", "max"]).reset_index())
    country_stats["cv"] = country_stats["std"] / country_stats["mean"].replace(0, np.nan)
    country_stats["size_log"] = np.log10(country_stats["mean"].clip(lower=1))

    # 3. CAGR histórico (1990 → 2019)
    cagr_hist = []
    for country, grp in country_yearly.groupby("Country"):
        grp = grp.sort_values("year")
        v = grp["consumption_cups"].values
        # Filtrar ceros iniciales
        nonzero = v > 0
        if not nonzero.any():
            cagr_hist.append({"Country": country, "growth_historical": np.nan})
            continue
        first_nz = np.argmax(nonzero)
        v_eff = v[first_nz:]
        n = len(v_eff) - 1
        if n < 5 or v_eff[0] <= 0:
            cagr_hist.append({"Country": country, "growth_historical": np.nan})
        else:
            cagr_hist.append({
                "Country": country,
                "growth_historical": ((v_eff[-1] / v_eff[0]) ** (1/n) - 1) * 100
            })
    cagr_hist_df = pd.DataFrame(cagr_hist)

    # 4. CAGR forecast (2019 → 2024) a nivel país
    fc_country = forecasts.groupby(["Country", "year"])["forecast"].sum().reset_index()
    actual_2019 = country_yearly[country_yearly["year"] == 2019].set_index("Country")["consumption_cups"]
    forecast_2024 = fc_country[fc_country["year"] == 2024].set_index("Country")["forecast"]

    growth_fc = pd.DataFrame({
        "actual_2019": actual_2019,
        "forecast_2024": forecast_2024,
    }).dropna()
    growth_fc["growth_forecast"] = ((growth_fc["forecast_2024"] / growth_fc["actual_2019"]) ** (1/5) - 1) * 100
    growth_fc = growth_fc.reset_index()[["Country", "growth_forecast"]]

    # 5. Predictibilidad: 100 - MAPE promedio del país (ponderado por consumo)
    bt_country = (bt_summary.merge(
        wide_df[["Country", "Coffee type", "mean_consumption"]],
        on=["Country", "Coffee type"], how="left"
    ))
    bt_country["weighted_mape"] = bt_country["best_mape"] * bt_country["mean_consumption"]
    pred_country = (bt_country.groupby("Country")
                    .apply(lambda g: g["weighted_mape"].sum() / g["mean_consumption"].sum()
                           if g["mean_consumption"].sum() > 0 else np.nan)
                    .reset_index(name="weighted_mape_country"))
    pred_country["predictability"] = (100 - pred_country["weighted_mape_country"]).clip(0, 100)
    pred_country = pred_country[["Country", "predictability"]]

    # 6. ¿Tuvo quiebre? (cualquier tipo de café del país)
    breaks_country = (breaks_df.groupby("Country")
                      .apply(lambda g: int(g["breakpoint_year"].notna().any()))
                      .reset_index(name="had_breakpoint"))

    # 7. Share global actual (2019)
    total_2019 = country_yearly[country_yearly["year"] == 2019]["consumption_cups"].sum()
    share_country = (country_yearly[country_yearly["year"] == 2019]
                     .groupby("Country")["consumption_cups"].sum() / total_2019 * 100).reset_index()
    share_country.columns = ["Country", "share_global_pct"]

    # 8. Región
    region_country = (long_df[["Country", "region"]].drop_duplicates()
                      .set_index("Country")["region"].reset_index())

    # MERGE TODO
    features = (country_stats[["Country", "mean", "cv", "size_log"]]
                .merge(cagr_hist_df, on="Country", how="left")
                .merge(growth_fc, on="Country", how="left")
                .merge(pred_country, on="Country", how="left")
                .merge(breaks_country, on="Country", how="left")
                .merge(share_country, on="Country", how="left")
                .merge(region_country, on="Country", how="left"))

    features["acceleration"] = features["growth_forecast"] - features["growth_historical"]

    return features


# ============================================================================
# CLUSTERING
# ============================================================================

# ============================================================================
# CLUSTERING
# ----------------------------------------------------------------------------
# DECISIÓN METODOLÓGICA: el clustering se hace SOBRE EL COMPORTAMIENTO
# (crecimiento, aceleración, volatilidad, predictibilidad), NO sobre el tamaño.
#
# Razón: Brasil concentra 44% del consumo global. Si incluimos size_log en el
# clustering, Brasil queda aislado en un clúster de un solo país, y los 54
# restantes terminan en un único cluster grande indistinguible.
#
# El tamaño es una dimensión IMPORTANTE — pero importa en el SCORING (IPS),
# no en el agrupamiento. Esta separación es estándar en analítica estratégica:
# "¿qué tipo de mercado es?" vs "¿qué tan grande es?" son preguntas distintas.
# ============================================================================
CLUSTER_FEATURES = ["growth_forecast", "acceleration", "cv", "predictability"]


def run_clustering(features, k_range=K_RANGE, random_state=RANDOM_STATE):
    """
    K-Means con selección de K por silhouette. Devuelve features con columna 'cluster'.

    Pipeline:
    1. Winsoriza features al 5%-95% para neutralizar outliers extremos
       (Venezuela en crash, Ghana en aceleración explosiva).
    2. Aplica RobustScaler (basado en mediana/IQR).
    3. Selecciona K óptimo por silhouette.
    4. Reduce a 2D con PCA para visualización.
    """
    df = features.dropna(subset=CLUSTER_FEATURES).copy()
    X = df[CLUSTER_FEATURES].values.copy()

    # WINSORIZACIÓN al 5%-95% por columna
    for j in range(X.shape[1]):
        col = X[:, j]
        p5, p95 = np.percentile(col, [5, 95])
        X[:, j] = np.clip(col, p5, p95)

    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Probar diferentes K
    silhouettes = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouettes[k] = sil

    best_k = max(silhouettes, key=silhouettes.get)

    # Fit final con el mejor K
    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=50)
    df["cluster"] = km_final.fit_predict(X_scaled)

    # PCA para visualización
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]

    # Centroides en espacio original (de-escalados)
    centroids_scaled = km_final.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_orig, columns=CLUSTER_FEATURES)
    centroids_df["cluster"] = range(best_k)
    centroids_df["n_countries"] = [
        (df["cluster"] == c).sum() for c in range(best_k)
    ]

    meta = {
        "best_k": int(best_k),
        "silhouette_scores": {int(k): float(v) for k, v in silhouettes.items()},
        "best_silhouette": float(silhouettes[best_k]),
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    }

    return df, centroids_df, meta


def name_clusters(centroids_df, country_segments_df):
    """
    Asigna nombre estratégico DETERMINISTA y ÚNICO a cada clúster.

    Lógica basada en thresholds explícitos sobre crecimiento y aceleración.
    Garantiza nombres únicos: si dos clústers caen en la misma categoría base,
    se diferencian por suffix.
    """
    names = {}
    descriptions = {}

    # Asignar categoría base por cada centroide
    base_categories = []
    for _, row in centroids_df.iterrows():
        c = int(row["cluster"])
        g = row["growth_forecast"]
        a = row["acceleration"]
        p = row["predictability"]

        if g < -8:
            base = ("Crisis Económica",
                    "Caídas severas de consumo. Posible crisis local. Liberan oferta exportable involuntariamente.")
        elif g < 0:
            base = ("Estancamiento",
                    "Consumo doméstico declinante leve. Mercados que pierden tracción.")
        elif a > 1.5:
            base = ("Aceleración Sostenida",
                    "Crecen y la velocidad sube. Máxima señal de tensión sobre oferta.")
        elif g > 2:
            base = ("Crecimiento Maduro",
                    "Crecen sólidamente pero el ritmo se está suavizando. Mercados consolidando.")
        elif g > 0.5:
            base = ("Maduros Estables",
                    "Crecimiento bajo y consistente. Base predecible de la demanda global.")
        else:
            base = ("Estables Predecibles",
                    "Consumo prácticamente constante. Comportamiento muy predecible.")

        base_categories.append((c, base[0], base[1], g, a, p))

    # Si hay nombres duplicados, diferenciar por intensidad
    name_counts = {}
    for c, name, _, _, _, _ in base_categories:
        name_counts[name] = name_counts.get(name, 0) + 1

    # Procesar y resolver duplicados
    seen_names = {}
    for c, base_name, desc, g, a, p in base_categories:
        if name_counts[base_name] > 1:
            # Diferenciar por intensidad: el de menor growth → "Severa", el otro → ""
            same_group = [(cc, gg) for cc, n, _, gg, _, _ in base_categories if n == base_name]
            same_group.sort(key=lambda x: x[1])  # menor growth primero
            rank = [i for i, (cc, _) in enumerate(same_group) if cc == c][0]
            if base_name == "Estancamiento":
                suffix = " Severo" if rank == 0 else " Leve"
            elif base_name == "Crecimiento Maduro":
                suffix = " Lento" if rank == 0 else " Rápido"
            else:
                suffix = f" (variante {rank+1})"
            final_name = base_name + suffix
        else:
            final_name = base_name

        names[c] = final_name
        descriptions[c] = desc

    return names, descriptions


# ============================================================================
# ÍNDICE DE PRESIÓN SOBRE OFERTA (IPS)
# ============================================================================

def compute_ips(features, weights=IPS_WEIGHTS):
    """
    Calcula el IPS para cada país.
    Normaliza cada componente al rango [0, 100] con MinMaxScaler.
    Componentes con valores bajos = menos presión → score bajo.
    Componentes con valores altos = más presión → score alto.
    """
    df = features.copy()
    components = list(weights.keys())

    # Normalizar componentes a 0-100
    for comp in components:
        col = df[comp]
        # Para predictability ya está en 0-100
        if comp == "predictability":
            df[f"{comp}_norm"] = col.fillna(col.median())
        else:
            scaler = MinMaxScaler(feature_range=(0, 100))
            valid = col.dropna()
            normalized = pd.Series(np.nan, index=df.index)
            if len(valid) > 0:
                normalized.loc[valid.index] = scaler.fit_transform(valid.values.reshape(-1, 1)).flatten()
            df[f"{comp}_norm"] = normalized.fillna(normalized.median())

    # Score compuesto
    df["ips"] = sum(df[f"{c}_norm"] * w for c, w in weights.items())

    # Componentes individuales para transparencia
    for comp in components:
        df[f"{comp}_contrib"] = df[f"{comp}_norm"] * weights[comp]

    return df


def assign_strategy(row):
    """
    Asigna recomendación comercial según el cuadrante en (IPS, predictabilidad).
    """
    ips = row["ips"]
    pred = row["predictability"]

    if ips > 60 and pred > 60:
        return "Priorizar — alta presión, alta confianza"
    elif ips > 60 and pred <= 60:
        return "Vigilar — alta presión pero forecast incierto"
    elif ips > 40 and pred > 60:
        return "Asegurar — presión media, predecible"
    elif ips > 40:
        return "Monitorear — presión media"
    elif pred > 70:
        return "Base estable — bajo riesgo de oferta"
    else:
        return "Baja prioridad"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    if (BASE.parent / "data").exists() and BASE.name == "src":
        BASE = BASE.parent
    DATA = BASE / "data"

    long_df = pd.read_parquet(DATA / "coffee_long.parquet")
    wide_df = pd.read_parquet(DATA / "coffee_wide.parquet")
    breaks_df = pd.read_parquet(DATA / "breakpoints.parquet")
    bt_summary = pd.read_parquet(DATA / "backtest_summary.parquet")
    forecasts = pd.read_parquet(DATA / "forecasts.parquet")

    print("=" * 70)
    print("MOTOR DE SEGMENTACIÓN Y SCORING")
    print("=" * 70)

    print("\n[1/4] Construyendo features por país...")
    features = build_country_features(long_df, wide_df, breaks_df, bt_summary, forecasts)
    print(f"  Países con features: {len(features)}")

    print("\n[2/4] Ejecutando clustering K-Means...")
    clustered, centroids, meta = run_clustering(features)
    print(f"  K óptimo: {meta['best_k']}  (silhouette = {meta['best_silhouette']:.3f})")
    print(f"  Silhouette por K: {meta['silhouette_scores']}")
    print(f"  Varianza explicada PCA: {sum(meta['explained_variance']):.1%}")

    print("\n[3/4] Nombrando clústers...")
    names, descriptions = name_clusters(centroids, clustered)
    clustered["cluster_name"] = clustered["cluster"].map(names)
    centroids["cluster_name"] = centroids["cluster"].map(names)

    print("\n  Clústers identificados:")
    for c in range(meta["best_k"]):
        n = (clustered["cluster"] == c).sum()
        print(f"    Cluster {c}: {names[c]}  ({n} países)")
        print(f"      └ {descriptions[c]}")

    print("\n[4/4] Calculando IPS y recomendaciones...")
    scored = compute_ips(clustered)
    scored["strategy"] = scored.apply(assign_strategy, axis=1)

    print(f"\nTop 10 países por IPS:")
    cols = ["Country", "region", "cluster_name", "ips", "strategy"]
    print(scored.nlargest(10, "ips")[cols].to_string(index=False))

    # Guardar
    scored.to_parquet(DATA / "country_segments.parquet", index=False)
    centroids.to_parquet(DATA / "cluster_centroids.parquet", index=False)

    with open(DATA / "clustering_meta.json", "w") as f:
        meta_out = {**meta, "cluster_names": {str(k): v for k, v in names.items()},
                    "cluster_descriptions": {str(k): v for k, v in descriptions.items()},
                    "ips_weights": IPS_WEIGHTS}
        json.dump(meta_out, f, indent=2)

    print(f"\n✓ Guardados en {DATA}")