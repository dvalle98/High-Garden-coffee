"""
Visualizaciones del Módulo 3 — Segmentación estratégica y scoring.
Cada figura responde a una pregunta de decisión comercial.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_style, title_block, footer, COLORS, CATEGORICAL, REGION_COLORS

BASE = Path(__file__).resolve().parent
if (BASE.parent / "data").exists() and BASE.name == "src":
    BASE = BASE.parent
DATA = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)


def load_all():
    segments = pd.read_parquet(DATA / "country_segments.parquet")
    centroids = pd.read_parquet(DATA / "cluster_centroids.parquet")
    long_df = pd.read_parquet(DATA / "coffee_long.parquet")
    return segments, centroids, long_df


# Paleta consistente para los 5 clústers (asignación dinámica luego)
CLUSTER_PALETTE = [
    COLORS["espresso"],
    COLORS["amber"],
    COLORS["olive"],
    COLORS["terracotta"],
    COLORS["teal"],
    COLORS["deep_red"],
    COLORS["gold"],
]


def fmt_billions(x, pos=None):
    if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6: return f"{x/1e6:.0f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def get_cluster_color_map(segments):
    """Asigna un color consistente a cada nombre de clúster."""
    unique_clusters = sorted(segments["cluster"].unique())
    return {int(c): CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, c in enumerate(unique_clusters)}


# ============================================================================
# FIGURA 13 — ESPACIO DE FEATURES (PCA)
# Tesis: los 5 clústers son separables en el espacio reducido. La segmentación
# no es arbitraria; emerge de la estructura natural de los datos.
# ============================================================================
def fig13_pca_space(segments, centroids):
    color_map = get_cluster_color_map(segments)

    fig = plt.figure(figsize=(13, 8))
    title_block(
        fig,
        eyebrow="SEGMENTACIÓN  ·  ESPACIO DE FEATURES (PCA 2D)",
        title="Cinco segmentos estratégicos emergen del comportamiento de los mercados",
        subtitle="Cada punto es un país, proyectado en 2D vía PCA sobre crecimiento, aceleración, volatilidad y predictibilidad. Los clústers están separados — no son artificio.",
    )

    ax = fig.add_axes([0.07, 0.13, 0.62, 0.70])

    # Plot cada clúster
    cluster_names = {}
    for cluster_id in sorted(segments["cluster"].unique()):
        sub = segments[segments["cluster"] == cluster_id]
        name = sub["cluster_name"].iloc[0]
        cluster_names[cluster_id] = name
        ax.scatter(sub["pca_x"], sub["pca_y"],
                   s=80, c=color_map[cluster_id], alpha=0.75,
                   edgecolors=COLORS["ink"], linewidths=0.6,
                   label=f"{name} ({len(sub)})", zorder=3)

    # Etiquetar países notables: top 5 IPS + outliers (top 3 PC1 absoluto)
    top_ips = segments.nlargest(5, "ips")
    extremes_x = pd.concat([segments.nlargest(2, "pca_x"), segments.nsmallest(2, "pca_x")])
    extremes_y = pd.concat([segments.nlargest(2, "pca_y"), segments.nsmallest(2, "pca_y")])
    to_label = pd.concat([top_ips, extremes_x, extremes_y]).drop_duplicates(subset=["Country"])

    for _, r in to_label.iterrows():
        ax.annotate(r["Country"], xy=(r["pca_x"], r["pca_y"]),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=8.5, color=COLORS["ink"], family="serif", weight="bold")

    ax.set_xlabel("Componente Principal 1", fontsize=10)
    ax.set_ylabel("Componente Principal 2", fontsize=10)
    ax.axhline(0, color=COLORS["rule"], lw=0.5, ls=":")
    ax.axvline(0, color=COLORS["rule"], lw=0.5, ls=":")

    # Panel derecho: leyenda con conteos
    box = fig.add_axes([0.72, 0.13, 0.25, 0.70])
    box.axis("off")
    box.text(0, 1.0, "SEGMENTOS", fontsize=8.5, color=COLORS["amber"],
             weight="bold", family="sans-serif", transform=box.transAxes)

    # Ordenar por tamaño descendente
    cluster_sizes = segments.groupby(["cluster", "cluster_name"]).size().reset_index(name="n")
    cluster_sizes = cluster_sizes.sort_values("n", ascending=False)

    for i, (_, row) in enumerate(cluster_sizes.iterrows()):
        y = 0.92 - i * 0.15
        c = int(row["cluster"])
        box.add_patch(Rectangle((0, y - 0.05), 0.025, 0.10,
                                facecolor=color_map[c],
                                transform=box.transAxes, clip_on=False))
        box.text(0.05, y + 0.015, row["cluster_name"], fontsize=10, color=COLORS["ink"],
                 family="serif", weight="bold", transform=box.transAxes)
        box.text(0.05, y - 0.025, f"{row['n']} países", fontsize=8.5, color=COLORS["muted"],
                 family="sans-serif", transform=box.transAxes)

    footer(fig)
    out = FIG / "13_pca_space.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 14 — PERFILES DE CLÚSTER (RADAR / PARALLEL COORDS)
# Tesis: cada segmento tiene una identidad multidimensional clara.
# ============================================================================
def fig14_cluster_profiles(segments, centroids):
    color_map = get_cluster_color_map(segments)

    fig = plt.figure(figsize=(13, 7))
    title_block(
        fig,
        eyebrow="PERFILES  ·  IDENTIDAD MULTIDIMENSIONAL DE CADA CLÚSTER",
        title="Cada segmento tiene una firma distinta en las 4 dimensiones",
        subtitle="Coordenadas paralelas normalizadas (0-100): cada línea es el centroide de un clúster cruzando las 4 dimensiones del análisis.",
    )

    # Normalizar features 0-100 para visualización comparativa
    feature_cols = ["growth_forecast", "acceleration", "cv", "predictability"]
    feature_labels = ["Crecimiento\nproyectado", "Aceleración\nreciente", "Volatilidad", "Predictibilidad"]
    cent_norm = centroids.copy()
    for f in feature_cols:
        col = cent_norm[f]
        cent_norm[f] = (col - col.min()) / (col.max() - col.min()) * 100

    ax = fig.add_axes([0.08, 0.16, 0.72, 0.62])
    x_pos = np.arange(len(feature_cols))

    for _, row in cent_norm.iterrows():
        c = int(row["cluster"])
        name = row["cluster_name"]
        y_vals = row[feature_cols].values.astype(float)
        ax.plot(x_pos, y_vals, color=color_map[c], lw=2.5, marker="o",
                markersize=10, label=name, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_labels, fontsize=10)
    ax.set_ylabel("Valor relativo (0 = mínimo, 100 = máximo del grupo)", fontsize=9.5)
    ax.set_ylim(-10, 110)

    # Eliminar bordes laterales
    ax.spines["bottom"].set_color(COLORS["rule"])
    ax.grid(axis="x", alpha=0.3)

    # Panel derecho: leyenda
    legend_ax = fig.add_axes([0.82, 0.16, 0.16, 0.62])
    legend_ax.axis("off")

    cent_sorted = cent_norm.sort_values("growth_forecast", ascending=False)
    for i, (_, row) in enumerate(cent_sorted.iterrows()):
        y = 0.94 - i * 0.16
        c = int(row["cluster"])
        legend_ax.plot([0, 0.15], [y, y], color=color_map[c], lw=2.5,
                       transform=legend_ax.transAxes)
        legend_ax.scatter([0.075], [y], color=color_map[c], s=40,
                          transform=legend_ax.transAxes, zorder=5)
        # Texto del nombre del cluster
        name_text = row["cluster_name"]
        # Wrap si es muy largo
        if len(name_text) > 18:
            words = name_text.split()
            mid = len(words) // 2
            name_text = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        legend_ax.text(0.18, y, name_text, fontsize=8.5, color=COLORS["ink"],
                       family="serif", weight="bold", va="center",
                       transform=legend_ax.transAxes)

    footer(fig)
    out = FIG / "14_cluster_profiles.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 15 — MATRIZ ESTRATÉGICA 2x2
# Tesis: el mapa accionable. Crecimiento × Tamaño con clústers superpuestos.
# Esta es la gráfica que el área comercial puede pegar en una diapositiva.
# ============================================================================
def fig15_strategic_matrix(segments, long_df):
    color_map = get_cluster_color_map(segments)

    # Computar tamaño en valor absoluto (mean_consumption)
    country_size = long_df.groupby("Country")["consumption_cups"].mean().reset_index()
    country_size.columns = ["Country", "mean_consumption"]
    df = segments.merge(country_size, on="Country", how="left")

    fig = plt.figure(figsize=(14, 9))
    title_block(
        fig,
        eyebrow="MATRIZ ESTRATÉGICA  ·  CRECIMIENTO × TAMAÑO  ·  POR CLÚSTER",
        title="El mapa accionable: dónde está cada mercado en el plano comercial",
        subtitle="Cuadrante superior derecho: grandes y creciendo rápido = máxima presión sobre oferta. Color del punto: clúster estratégico (basado en comportamiento, no tamaño).",
    )

    ax = fig.add_axes([0.07, 0.10, 0.62, 0.72])

    # Líneas medianas
    median_size = df["mean_consumption"].median()
    median_growth = df["growth_forecast"].median()
    ax.axvline(median_size, color=COLORS["rule"], lw=0.8, ls="--")
    ax.axhline(median_growth, color=COLORS["rule"], lw=0.8, ls="--")

    # Etiquetas de cuadrantes
    ax.text(0.97, 0.97, "GRANDES & CRECIENDO\nmáxima presión",
            transform=ax.transAxes, ha="right", va="top",
            color=COLORS["deep_red"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.03, 0.97, "PEQUEÑOS & CRECIENDO\nemergentes a vigilar",
            transform=ax.transAxes, ha="left", va="top",
            color=COLORS["amber"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.97, 0.03, "GRANDES & ESTABLES\nbase predecible",
            transform=ax.transAxes, ha="right", va="bottom",
            color=COLORS["olive"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.03, 0.03, "PEQUEÑOS & ESTABLES\nbaja prioridad",
            transform=ax.transAxes, ha="left", va="bottom",
            color=COLORS["slate"], fontsize=9, weight="bold", family="sans-serif")

    # Scatter por clúster
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        ax.scatter(sub["mean_consumption"], sub["growth_forecast"],
                   s=80, c=color_map[cluster_id], alpha=0.75,
                   edgecolors=COLORS["ink"], linewidths=0.6, zorder=3)

    # Etiquetar países notables
    notable = pd.concat([
        df.nlargest(10, "mean_consumption"),  # top 10 grandes
        df.nlargest(5, "growth_forecast"),     # top 5 crecimiento
        df.nsmallest(3, "growth_forecast"),    # bottom 3 decline
    ]).drop_duplicates(subset=["Country"])

    for _, r in notable.iterrows():
        ax.annotate(r["Country"],
                    xy=(r["mean_consumption"], r["growth_forecast"]),
                    xytext=(7, 5), textcoords="offset points",
                    fontsize=8.5, color=COLORS["ink"],
                    family="serif", weight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Tamaño del mercado (consumo promedio, tazas, log)", fontsize=10)
    ax.set_ylabel("Crecimiento proyectado 2019-2024 (% anual)", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_billions))

    # Panel derecho: playbook por cluster
    box = fig.add_axes([0.72, 0.10, 0.26, 0.72])
    box.axis("off")
    box.text(0, 1.0, "PLAYBOOK POR SEGMENTO", fontsize=8.5, color=COLORS["amber"],
             weight="bold", family="sans-serif", transform=box.transAxes)

    cluster_playbooks = {
        "Aceleración Sostenida": "Asegurar abastecimiento a largo plazo. Riesgo: que el productor priorice mercado interno.",
        "Crecimiento Maduro": "Mantener relaciones. Volumen estable a corto plazo.",
        "Maduros Estables": "Aliados confiables. Buenos para contratos plurianuales.",
        "Estables Predecibles": "Base operativa. Volumen consolidado y predecible.",
        "Estancamiento Leve": "Posible oportunidad: oferta exportable creciente.",
        "Estancamiento Severo": "Oferta liberada involuntariamente. Riesgo geopolítico.",
    }

    cluster_sizes = df.groupby(["cluster", "cluster_name"]).size().reset_index(name="n")
    cluster_sizes = cluster_sizes.sort_values("n", ascending=False)

    for i, (_, row) in enumerate(cluster_sizes.iterrows()):
        y = 0.92 - i * 0.17
        c = int(row["cluster"])
        playbook = cluster_playbooks.get(row["cluster_name"], "")
        box.add_patch(Rectangle((0, y - 0.05), 0.025, 0.10,
                                facecolor=color_map[c],
                                transform=box.transAxes, clip_on=False))
        box.text(0.05, y + 0.025, row["cluster_name"], fontsize=9.5, color=COLORS["ink"],
                 family="serif", weight="bold", transform=box.transAxes)
        box.text(0.05, y - 0.005, f"{row['n']} países", fontsize=7.5,
                 color=COLORS["muted"], family="sans-serif", transform=box.transAxes)
        # Playbook text con wrap
        words = playbook.split()
        line1 = ""
        line2 = ""
        for w in words:
            if len(line1) + len(w) < 32:
                line1 += w + " "
            else:
                line2 += w + " "
        box.text(0.05, y - 0.035, line1.strip(), fontsize=7,
                 color=COLORS["muted"], family="serif", style="italic",
                 transform=box.transAxes)
        if line2.strip():
            box.text(0.05, y - 0.055, line2.strip(), fontsize=7,
                     color=COLORS["muted"], family="serif", style="italic",
                     transform=box.transAxes)

    footer(fig)
    out = FIG / "15_strategic_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 16 — LEADERBOARD IPS COMPLETO
# Tesis: ranking final accionable de los 55 países, ordenado por IPS,
# con descomposición de los 4 componentes del score.
# ============================================================================
def fig16_ips_leaderboard(segments):
    color_map = get_cluster_color_map(segments)

    df = segments.sort_values("ips", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(13, 12))
    title_block(
        fig,
        eyebrow="LEADERBOARD  ·  ÍNDICE DE PRESIÓN SOBRE OFERTA (IPS)",
        title="Ranking final: dónde va a sentirse más la tensión sobre la oferta exportable",
        subtitle="IPS = 30%·tamaño + 30%·crecimiento + 20%·aceleración + 20%·predictibilidad. Cada barra apilada muestra la contribución de cada componente.",
    )

    ax = fig.add_axes([0.16, 0.04, 0.75, 0.80])

    # Stacked bars con los 4 componentes
    contribs = ["size_log_contrib", "growth_forecast_contrib",
                "acceleration_contrib", "predictability_contrib"]
    contrib_labels = ["Tamaño", "Crecimiento", "Aceleración", "Predictibilidad"]
    contrib_colors = [COLORS["espresso"], COLORS["amber"],
                      COLORS["olive"], COLORS["teal"]]

    y_pos = np.arange(len(df))
    left = np.zeros(len(df))
    for col, label, color in zip(contribs, contrib_labels, contrib_colors):
        vals = df[col].values
        ax.barh(y_pos, vals, left=left, color=color, alpha=0.85,
                edgecolor=COLORS["bg"], lw=0.3, label=label)
        left += vals

    # Etiqueta de país en y, IPS total al final
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Country"], fontsize=7.5)
    ax.tick_params(axis="y", length=0)

    for i, ips in enumerate(df["ips"]):
        ax.text(ips + 0.5, i, f"{ips:.1f}",
                fontsize=7.5, va="center", color=COLORS["ink"],
                family="serif", weight="bold")

    ax.set_xlabel("IPS = suma de las 4 contribuciones (0–100)", fontsize=9.5)
    ax.set_xlim(0, 105)

    # Leyenda horizontal arriba
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.04),
              ncol=4, fontsize=9, frameon=False)

    footer(fig)
    out = FIG / "16_ips_leaderboard.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 17 — TARJETAS DEL TOP 5
# Tesis: perfil detallado de los 5 países que High Garden debe priorizar.
# ============================================================================
def fig17_top5_profiles(segments, long_df):
    top5 = segments.nlargest(5, "ips").reset_index(drop=True)
    color_map = get_cluster_color_map(segments)

    fig = plt.figure(figsize=(14, 8.5))
    title_block(
        fig,
        eyebrow="TOP 5 PRIORIDADES  ·  PERFIL ESTRATÉGICO DETALLADO",
        title="Los cinco países con mayor presión sobre la oferta exportable",
        subtitle="Cada tarjeta resume el perfil del país: tamaño, crecimiento, segmento, predictibilidad y recomendación.",
    )

    for i, (_, row) in enumerate(top5.iterrows()):
        col = i
        ax = fig.add_axes([0.05 + col * 0.19, 0.18, 0.16, 0.62])
        ax.axis("off")
        c = int(row["cluster"])
        accent = color_map[c]

        # Caja de fondo
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                     boxstyle="round,pad=0.02,rounding_size=0.04",
                                     facecolor=COLORS["bg"],
                                     edgecolor=accent, linewidth=2,
                                     transform=ax.transAxes))

        # Banner top con el clúster
        ax.add_patch(Rectangle((0.02, 0.86), 0.96, 0.12,
                               facecolor=accent, alpha=0.85,
                               transform=ax.transAxes))
        ax.text(0.5, 0.92, f"#{i+1}  ·  IPS {row['ips']:.1f}",
                fontsize=11, color=COLORS["bg"], weight="bold",
                family="serif", ha="center", transform=ax.transAxes)

        # Nombre del país
        ax.text(0.5, 0.78, row["Country"], fontsize=14, color=COLORS["ink"],
                weight="bold", family="serif", ha="center",
                transform=ax.transAxes)
        ax.text(0.5, 0.73, row["region"], fontsize=8.5, color=COLORS["muted"],
                family="sans-serif", ha="center", style="italic",
                transform=ax.transAxes)

        # Segmento
        ax.text(0.5, 0.66, row["cluster_name"].upper(), fontsize=8,
                color=accent, weight="bold", family="sans-serif",
                ha="center", transform=ax.transAxes)

        # Métricas
        country_hist = long_df[long_df["Country"] == row["Country"]]
        size_2019 = country_hist[country_hist["year"] == 2019]["consumption_cups"].sum() / 1e6

        metrics = [
            ("Consumo 2019",   f"{size_2019:.0f}M"),
            ("Share global",   f"{row['share_global_pct']:.1f}%"),
            ("CAGR proyectado", f"{row['growth_forecast']:+.1f}%"),
            ("Aceleración",     f"{row['acceleration']:+.1f} pp"),
            ("Predictibilidad", f"{row['predictability']:.0f}/100"),
        ]
        y_start = 0.58
        for j, (label, value) in enumerate(metrics):
            y = y_start - j * 0.07
            ax.text(0.08, y, label, fontsize=7.5, color=COLORS["muted"],
                    family="sans-serif", weight="bold", transform=ax.transAxes)
            ax.text(0.92, y, value, fontsize=8.5, color=COLORS["ink"],
                    family="serif", weight="bold", ha="right",
                    transform=ax.transAxes)

        # Estrategia
        ax.text(0.5, 0.16, "ESTRATEGIA", fontsize=7, color=COLORS["amber"],
                weight="bold", family="sans-serif", ha="center",
                transform=ax.transAxes)
        # Wrap strategy text
        strategy = row["strategy"]
        if "—" in strategy:
            parts = strategy.split("—")
            ax.text(0.5, 0.10, parts[0].strip(), fontsize=9, color=COLORS["espresso"],
                    weight="bold", family="serif", ha="center",
                    transform=ax.transAxes)
            ax.text(0.5, 0.05, parts[1].strip(), fontsize=7, color=COLORS["muted"],
                    family="serif", ha="center", style="italic",
                    transform=ax.transAxes)
        else:
            ax.text(0.5, 0.07, strategy, fontsize=8.5, color=COLORS["espresso"],
                    weight="bold", family="serif", ha="center",
                    transform=ax.transAxes)

    footer(fig)
    out = FIG / "17_top5_profiles.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# MAIN
# ============================================================================
def main():
    apply_style()
    segments, centroids, long_df = load_all()
    print("Construyendo visualizaciones del Módulo 3...\n")

    outputs = []
    outputs.append(fig13_pca_space(segments, centroids))
    print("  ✓ 13 — Espacio PCA")
    outputs.append(fig14_cluster_profiles(segments, centroids))
    print("  ✓ 14 — Perfiles de clúster")
    outputs.append(fig15_strategic_matrix(segments, long_df))
    print("  ✓ 15 — Matriz estratégica")
    outputs.append(fig16_ips_leaderboard(segments))
    print("  ✓ 16 — Leaderboard IPS")
    outputs.append(fig17_top5_profiles(segments, long_df))
    print("  ✓ 17 — Perfiles top 5")

    print(f"\nGuardadas {len(outputs)} figuras en {FIG}/")
    return outputs


if __name__ == "__main__":
    main()