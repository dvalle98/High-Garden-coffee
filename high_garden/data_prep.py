"""
High Garden Coffee — Data Preparation Module
=============================================
Carga, valida y enriquece el dataset de consumo doméstico de café (ICO).

Salidas:
    - data/coffee_long.parquet      : formato long (país, tipo, año, consumo)
    - data/coffee_wide.parquet      : formato wide enriquecido
    - data/validation_report.json   : reporte de validaciones
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
# Detectar si estamos en notebook o script para obtener raiz del proyecto
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    # Estamos en un notebook Jupyter
    BASE_DIR = Path.cwd().parent
RAW_PATH = BASE_DIR / "data" / "coffee_db.parquet"
OUT_DIR = BASE_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo país → región/subregión.
# Basado en la composición geográfica del cinturón cafetero.
REGION_MAP = {
    # SUR AMÉRICA
    "Brazil": ("South America", "Brazil"),
    "Colombia": ("South America", "Andean"),
    "Ecuador": ("South America", "Andean"),
    "Peru": ("South America", "Andean"),
    "Venezuela": ("South America", "Andean"),
    "Bolivia (Plurinational State of)": ("South America", "Andean"),
    "Paraguay": ("South America", "Southern Cone"),
    "Guyana": ("South America", "Guianas"),
    # CENTRO AMÉRICA Y CARIBE
    "Mexico": ("Central America", "Mexico"),
    "Guatemala": ("Central America", "CA-Norte"),
    "Honduras": ("Central America", "CA-Norte"),
    "El Salvador": ("Central America", "CA-Norte"),
    "Nicaragua": ("Central America", "CA-Sur"),
    "Costa Rica": ("Central America", "CA-Sur"),
    "Panama": ("Central America", "CA-Sur"),
    "Cuba": ("Caribbean", "Greater Antilles"),
    "Dominican Republic": ("Caribbean", "Greater Antilles"),
    "Haiti": ("Caribbean", "Greater Antilles"),
    "Jamaica": ("Caribbean", "Greater Antilles"),
    "Trinidad & Tobago": ("Caribbean", "Lesser Antilles"),
    # ÁFRICA
    "Ethiopia": ("Africa", "East Africa"),
    "Kenya": ("Africa", "East Africa"),
    "Tanzania": ("Africa", "East Africa"),
    "Uganda": ("Africa", "East Africa"),
    "Rwanda": ("Africa", "East Africa"),
    "Burundi": ("Africa", "East Africa"),
    "Madagascar": ("Africa", "East Africa"),
    "Malawi": ("Africa", "East Africa"),
    "Zambia": ("Africa", "East Africa"),
    "Zimbabwe": ("Africa", "East Africa"),
    "Cameroon": ("Africa", "Central Africa"),
    "Central African Republic": ("Africa", "Central Africa"),
    "Congo": ("Africa", "Central Africa"),
    "Democratic Republic of Congo": ("Africa", "Central Africa"),
    "Gabon": ("Africa", "Central Africa"),
    "Equatorial Guinea": ("Africa", "Central Africa"),
    "Angola": ("Africa", "Southern Africa"),
    "Côte d'Ivoire": ("Africa", "West Africa"),
    "Ghana": ("Africa", "West Africa"),
    "Guinea": ("Africa", "West Africa"),
    "Liberia": ("Africa", "West Africa"),
    "Sierra Leone": ("Africa", "West Africa"),
    "Nigeria": ("Africa", "West Africa"),
    "Togo": ("Africa", "West Africa"),
    # ASIA
    "Viet Nam": ("Asia", "Southeast Asia"),
    "Indonesia": ("Asia", "Southeast Asia"),
    "Thailand": ("Asia", "Southeast Asia"),
    "Philippines": ("Asia", "Southeast Asia"),
    "Lao People's Democratic Republic": ("Asia", "Southeast Asia"),
    "Timor-Leste": ("Asia", "Southeast Asia"),
    "Papua New Guinea": ("Asia", "Oceania"),
    "India": ("Asia", "South Asia"),
    "Sri Lanka": ("Asia", "South Asia"),
    "Nepal": ("Asia", "South Asia"),
    "Yemen": ("Asia", "Middle East"),
}


def load_raw():
    """Carga el dataset original."""
    df = pd.read_parquet(RAW_PATH)
    df.columns = df.columns.str.strip()
    return df


def get_year_cols(df):
    """Identifica columnas de años (formato coffee year 'YYYY/YY')."""
    return [c for c in df.columns if "/" in c]


def validate(df):
    """
    Valida la integridad estructural del dataset.
    Retorna un diccionario con todos los chequeos.
    """
    year_cols = get_year_cols(df)
    report = {}

    # 1. Estructura básica
    report["shape"] = df.shape
    report["n_countries"] = df["Country"].nunique()
    report["n_years"] = len(year_cols)
    report["period"] = f"{year_cols[0]} → {year_cols[-1]}"
    report["coffee_types"] = df["Coffee type"].value_counts().to_dict()

    # 2. Nulos
    report["nulls"] = df.isnull().sum().sum()

    # 3. Duplicados de país-tipo
    dup = df.duplicated(subset=["Country", "Coffee type"]).sum()
    report["duplicates_country_type"] = int(dup)

    # 4. Validación cruzada: ¿la columna Total_domestic_consumption coincide
    #    con la suma de los años anuales?
    computed_total = df[year_cols].sum(axis=1)
    given_total = df["Total_domestic_consumption"]
    diff_pct = ((computed_total - given_total).abs() / given_total.replace(0, np.nan) * 100)
    report["total_consistency"] = {
        "max_pct_diff": float(diff_pct.max()),
        "n_inconsistent_above_1pct": int((diff_pct > 1).sum()),
    }

    # 5. Ceros: ¿cuántos países tienen ceros en al menos un año?
    has_zero = (df[year_cols] == 0).any(axis=1)
    report["countries_with_zeros"] = {
        "count": int(has_zero.sum()),
        "countries": df.loc[has_zero, "Country"].tolist(),
    }

    # 6. Coverage de mapeo de regiones
    unmapped = [c for c in df["Country"].unique() if c not in REGION_MAP]
    report["unmapped_countries"] = unmapped

    return report


def to_long(df):
    """Convierte de wide a long format."""
    year_cols = get_year_cols(df)
    long = df.melt(
        id_vars=["Country", "Coffee type"],
        value_vars=year_cols,
        var_name="coffee_year",
        value_name="consumption_cups",
    )
    # El "coffee year" 1990/91 lo representamos por su año inicial 1990
    long["year"] = long["coffee_year"].str[:4].astype(int)
    long["consumption_cups"] = long["consumption_cups"].astype(float)
    return long


def enrich(long, wide):
    """Añade variables derivadas útiles para EDA y modelado."""
    # Región y subregión
    long["region"] = long["Country"].map(lambda c: REGION_MAP.get(c, ("Unknown", "Unknown"))[0])
    long["subregion"] = long["Country"].map(lambda c: REGION_MAP.get(c, ("Unknown", "Unknown"))[1])

    # YoY growth por país
    long = long.sort_values(["Country", "year"])
    long["yoy_growth_pct"] = (
        long.groupby("Country")["consumption_cups"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan) * 100
    )

    # Wide enriquecido: agregados por país
    year_cols = get_year_cols(wide)
    wide = wide.copy()
    wide["region"] = wide["Country"].map(lambda c: REGION_MAP.get(c, ("Unknown", "Unknown"))[0])
    wide["subregion"] = wide["Country"].map(lambda c: REGION_MAP.get(c, ("Unknown", "Unknown"))[1])

    # Métricas resumen
    wide["mean_consumption"] = wide[year_cols].mean(axis=1)
    wide["std_consumption"] = wide[year_cols].std(axis=1)
    wide["cv_consumption"] = wide["std_consumption"] / wide["mean_consumption"].replace(0, np.nan)

    # CAGR primera mitad vs segunda mitad (1990-2005 vs 2005-2020)
    half = len(year_cols) // 2
    start = wide[year_cols[0]].replace(0, np.nan)
    mid = wide[year_cols[half]].replace(0, np.nan)
    end = wide[year_cols[-1]].replace(0, np.nan)
    n_first = half
    n_second = len(year_cols) - 1 - half
    wide["cagr_1990_2005"] = ((mid / start) ** (1 / n_first) - 1) * 100
    wide["cagr_2005_2020"] = ((end / mid) ** (1 / n_second) - 1) * 100
    wide["cagr_full"] = ((end / start) ** (1 / (len(year_cols) - 1)) - 1) * 100

    # Share global por año inicial y final
    wide["share_1990"] = wide[year_cols[0]] / wide[year_cols[0]].sum() * 100
    wide["share_2020"] = wide[year_cols[-1]] / wide[year_cols[-1]].sum() * 100
    wide["share_change"] = wide["share_2020"] - wide["share_1990"]

    return long, wide


def main():
    print("=" * 70)
    print("HIGH GARDEN COFFEE — Data Preparation")
    print("=" * 70)

    df = load_raw()
    print(f"\n[1/4] Loaded raw data: {df.shape}")

    report = validate(df)
    print("\n[2/4] Validation report:")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    long = to_long(df)
    long, wide = enrich(long, df)
    print(f"\n[3/4] Enriched data:")
    print(f"  long: {long.shape}  cols: {long.columns.tolist()}")
    print(f"  wide: {wide.shape}")

    long.to_parquet(OUT_DIR / "coffee_long.parquet", index=False)
    wide.to_parquet(OUT_DIR / "coffee_wide.parquet", index=False)
    with open(OUT_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[4/4] Outputs saved to {OUT_DIR}")
    print("✓ Done")


if __name__ == "__main__":
    main()