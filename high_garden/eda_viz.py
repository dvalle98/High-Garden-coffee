"""
EDA visualizations — Módulo 1
Cada figura tiene una tesis explícita.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_style, title_block, footer, COLORS, CATEGORICAL, REGION_COLORS

BASE = Path(__file__).resolve().parent
if (BASE.parent / "data").exists() and BASE.name == "src":
    BASE = BASE.parent
DATA = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)


def load_data():
    long = pd.read_parquet(DATA / "coffee_long.parquet")
    wide = pd.read_parquet(DATA / "coffee_wide.parquet")
    return long, wide


def fmt_billions(x, pos=None):
    if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6: return f"{x/1e6:.0f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# ============================================================================
# FIGURA 1 — LA HISTORIA MACRO
# Tesis: el consumo doméstico de productores creció +156% en 30 años,
# señal estructural de presión sobre la oferta exportable global.
# ============================================================================
def fig1_macro_story(long):
    yearly = long.groupby("year")["consumption_cups"].sum().reset_index()
    yearly["yoy"] = yearly["consumption_cups"].pct_change() * 100

    fig = plt.figure(figsize=(13, 7.5))
    title_block(
        fig,
        eyebrow="MACRO  ·  1990 — 2020",
        title="El consumo interno de los países productores se multiplicó por 2.6",
        subtitle="Cada taza retenida internamente es una taza menos disponible para exportación. Hay menos café \"libre\" en el mercado global hoy que hace 30 años.",
    )

    # Panel principal
    ax = fig.add_axes([0.06, 0.18, 0.62, 0.62])
    ax.plot(yearly["year"], yearly["consumption_cups"],
            color=COLORS["espresso"], linewidth=2.3, zorder=3)
    ax.fill_between(yearly["year"], 0, yearly["consumption_cups"],
                    color=COLORS["espresso"], alpha=0.08, zorder=1)

    # Hitos
    milestones = [
        (2008, "Crisis financiera global"),
        (2014, "Roya del café (LATAM)"),
        (2019, "COVID-19  →  primera caída anual"),
    ]
    for yr, label in milestones:
        if yr in yearly["year"].values:
            v = yearly.loc[yearly["year"] == yr, "consumption_cups"].values[0]
            ax.axvline(yr, color=COLORS["muted"], lw=0.5, ls=":", alpha=0.6, zorder=2)
            ax.scatter([yr], [v], s=22, color=COLORS["terracotta"], zorder=4)
            ax.annotate(label, xy=(yr, v), xytext=(yr-2, v*1.07),
                        fontsize=8.5, color=COLORS["ink"], family="serif",
                        ha="right")

    # Endpoints
    y0, y1 = yearly["consumption_cups"].iloc[0], yearly["consumption_cups"].iloc[-1]
    ax.scatter([1990], [y0], color=COLORS["espresso"], s=42, zorder=5)
    ax.scatter([2019], [y1], color=COLORS["espresso"], s=42, zorder=5)
    ax.text(1990.3, y0 - 1.5e8, f"{fmt_billions(y0)}\n1990/91", fontsize=9,
            color=COLORS["muted"], family="serif")
    ax.text(2019.3, y1, f"  {fmt_billions(y1)}\n  2019/20", fontsize=9,
            color=COLORS["espresso"], family="serif", weight="bold")

    ax.set_ylabel("Consumo anual (tazas)", fontsize=10)
    ax.set_xlabel("")
    ax.set_xlim(1989, 2022)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_billions))

    # Panel lateral: KPIs editoriales
    n = len(yearly) - 1
    cagr = ((y1 / y0) ** (1 / n) - 1) * 100

    box = fig.add_axes([0.72, 0.18, 0.23, 0.62])
    box.axis("off")
    kpis = [
        ("CRECIMIENTO TOTAL",  f"+{(y1/y0-1)*100:.0f}%",  "30 años"),
        ("CAGR",                f"+{cagr:.2f}%",          "compuesto anual"),
        ("CONSUMO 1990",        fmt_billions(y0),         "tazas"),
        ("CONSUMO 2020",        fmt_billions(y1),         "tazas"),
        ("PRIMERA CAÍDA",       "2019/20",                f"{yearly['yoy'].iloc[-1]:.2f}% YoY"),
    ]
    for i, (label, value, sub) in enumerate(kpis):
        y = 0.92 - i*0.18
        box.text(0, y,        label, fontsize=8, color=COLORS["amber"],
                 weight="bold", family="sans-serif", transform=box.transAxes)
        box.text(0, y-0.045,  value, fontsize=20, color=COLORS["espresso"],
                 weight="bold", family="serif", transform=box.transAxes)
        box.text(0, y-0.085,  sub,   fontsize=8.5, color=COLORS["muted"],
                 family="serif", transform=box.transAxes)

    footer(fig)
    out = FIG / "01_macro_story.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 2 — LA HISTORIA DE LA CONCENTRACIÓN
# Tesis: a pesar del "boom emergente", el mercado se está concentrando.
# Brasil amplió su dominancia. Diversificar el origen es más difícil hoy.
# ============================================================================
def fig2_concentration(long, wide):
    year_cols = [c for c in wide.columns if "/" in c]

    # Share por año, top 8 + "Resto"
    yearly_country = long.pivot_table(
        index="year", columns="Country", values="consumption_cups", aggfunc="sum"
    )
    totals = yearly_country.sum(axis=1)
    shares = yearly_country.div(totals, axis=0) * 100

    avg_share = shares.mean().sort_values(ascending=False)
    top8 = avg_share.head(8).index.tolist()
    shares_plot = shares[top8].copy()
    shares_plot["Resto"] = 100 - shares_plot.sum(axis=1)

    fig = plt.figure(figsize=(13, 7.8))
    title_block(
        fig,
        eyebrow="CONCENTRACIÓN  ·  ¿SE DIVERSIFICÓ EL MERCADO?",
        title="El mercado se está concentrando, no diversificando",
        subtitle="Brasil pasó del 42% al 44% del consumo doméstico mundial. El \"boom emergente\" no diversificó el mercado: lo intensificó.",
    )

    # Panel izquierdo: stacked area de shares
    ax = fig.add_axes([0.06, 0.15, 0.50, 0.66])
    colors_stack = CATEGORICAL[:8] + [COLORS["rule"]]
    ax.stackplot(shares_plot.index, shares_plot.T.values,
                 labels=shares_plot.columns, colors=colors_stack,
                 alpha=0.92, edgecolor=COLORS["bg"], linewidth=0.4)
    ax.set_ylim(0, 100)
    ax.set_xlim(1990, 2019)
    ax.set_ylabel("Participación en consumo global (%)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Etiquetas en panel separado a la derecha del stacked
    # Algoritmo simple: si dos etiquetas están muy cerca, las separo verticalmente
    label_ax = fig.add_axes([0.57, 0.15, 0.14, 0.66])
    label_ax.axis("off")
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(0, 100)
    last_year = shares_plot.index.max()

    # Calcular posiciones reales (centro de cada banda)
    positions = []
    cum = 0
    for country, color in zip(shares_plot.columns, colors_stack):
        val = shares_plot.loc[last_year, country]
        center = cum + val/2
        positions.append({"country": country, "color": color, "val": val,
                          "real_y": center, "label_y": center})
        cum += val

    # Separar etiquetas que se solapan (espacio mínimo 3.5 puntos)
    min_gap = 3.5
    for i in range(1, len(positions)):
        if positions[i]["label_y"] - positions[i-1]["label_y"] < min_gap:
            positions[i]["label_y"] = positions[i-1]["label_y"] + min_gap

    for p in positions:
        if p["val"] > 0.5:
            # Línea conectora desde el centro REAL al centro de la etiqueta
            label_ax.plot([0, 0.2, 0.25],
                          [p["real_y"], p["real_y"], p["label_y"]],
                          color=p["color"], lw=1, alpha=0.7)
            label_ax.text(0.32, p["label_y"], p["country"], fontsize=8.5,
                          color=COLORS["ink"], family="serif", va="center")
            label_ax.text(0.32, p["label_y"] - 2.2, f"{p['val']:.1f}%", fontsize=7.5,
                          color=COLORS["muted"], family="sans-serif", va="center")

    # Panel derecho: HHI
    ax2 = fig.add_axes([0.74, 0.15, 0.22, 0.58])
    sh_prop = yearly_country.div(totals, axis=0)
    hhi = (sh_prop**2).sum(axis=1) * 10000

    ax2.plot(hhi.index, hhi.values, color=COLORS["terracotta"], lw=2.3)
    ax2.fill_between(hhi.index, 1900, hhi.values, color=COLORS["terracotta"], alpha=0.1)
    ax2.scatter([hhi.index[0], hhi.index[-1]], [hhi.iloc[0], hhi.iloc[-1]],
                color=COLORS["terracotta"], s=48, zorder=5)

    # Etiquetas de endpoints — separadas claramente
    ax2.annotate(f"{hhi.iloc[0]:.0f}", xy=(hhi.index[0], hhi.iloc[0]),
                 xytext=(8, -15), textcoords="offset points",
                 fontsize=11, color=COLORS["terracotta"], weight="bold",
                 family="serif")
    ax2.annotate(f"{hhi.iloc[-1]:.0f}", xy=(hhi.index[-1], hhi.iloc[-1]),
                 xytext=(-22, 10), textcoords="offset points",
                 fontsize=11, color=COLORS["terracotta"], weight="bold",
                 family="serif")

    # Cambio neto destacado — encima del panel HHI
    delta = hhi.iloc[-1] - hhi.iloc[0]
    fig.text(0.85, 0.79, "Índice Herfindahl-Hirschman", fontsize=10.5,
             color=COLORS["ink"], weight="bold", family="serif", ha="center")
    fig.text(0.85, 0.762, f"Δ HHI: +{delta:.0f} pts  (más concentrado)",
             fontsize=9, color=COLORS["deep_red"], weight="bold",
             family="serif", ha="center")

    ax2.set_ylabel("HHI", fontsize=9)
    ax2.set_ylim(1900, 2700)
    ax2.set_xlim(1990, 2019)

    footer(fig)
    out = FIG / "02_concentration.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 3 — EL CUADRANTE DE ACELERACIÓN (LA GRÁFICA "MONEY")
# Tesis: clasifica cada país según su trayectoria. Identifica los mercados
# emergentes reales, los maduros estables, y los que están perdiendo terreno.
# ============================================================================
def fig3_acceleration_quadrant(wide):
    df = wide[wide["mean_consumption"] > 500_000].copy()
    df = df.dropna(subset=["cagr_1990_2005", "cagr_2005_2020"])

    fig = plt.figure(figsize=(13, 8))
    title_block(
        fig,
        eyebrow="ACELERACIÓN  ·  MAPA ESTRATÉGICO DE MERCADOS",
        title="Cada país, dos trayectorias: cómo creció antes vs. cómo crece hoy",
        subtitle="El cuadrante superior derecho contiene los mercados consistentemente fuertes. El inferior derecho, los que se enfriaron. Indicador clave para asignar esfuerzo comercial.",
    )

    ax = fig.add_axes([0.08, 0.13, 0.84, 0.71])

    # Ejes de referencia
    ax.axhline(0, color=COLORS["rule"], lw=1, zorder=1)
    ax.axvline(0, color=COLORS["rule"], lw=1, zorder=1)

    # Sombras de cuadrantes con etiquetas
    quad_style = dict(fontsize=9, color=COLORS["muted"], style="italic",
                      family="serif", alpha=0.85)
    ax.text(0.97, 0.97, "ACELERACIÓN SOSTENIDA\ncrecieron antes  ·  siguen creciendo",
            transform=ax.transAxes, ha="right", va="top",
            color=COLORS["olive"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.03, 0.97, "RECIÉN ACELERANDO\nestancados antes  ·  crecen ahora",
            transform=ax.transAxes, ha="left", va="top",
            color=COLORS["amber"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.97, 0.03, "DESACELERANDO\ncrecieron antes  ·  hoy planos",
            transform=ax.transAxes, ha="right", va="bottom",
            color=COLORS["terracotta"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.03, 0.03, "ESTRUCTURALMENTE DÉBILES\nestancados antes y ahora",
            transform=ax.transAxes, ha="left", va="bottom",
            color=COLORS["deep_red"], fontsize=9, weight="bold", family="sans-serif")

    # Tamaño = consumo promedio (escala log para que sea legible)
    sizes = np.sqrt(df["mean_consumption"]) / 15

    # Colorear por región
    colors = df["region"].map(REGION_COLORS).fillna(COLORS["muted"])

    ax.scatter(df["cagr_1990_2005"], df["cagr_2005_2020"],
               s=sizes, c=colors, alpha=0.55,
               edgecolors=COLORS["ink"], linewidths=0.5, zorder=3)

    # Etiquetar los más relevantes
    to_label = pd.concat([
        df.nlargest(6, "mean_consumption"),       # los más grandes
        df.nlargest(4, "cagr_2005_2020"),         # los que más crecen ahora
        df.nsmallest(3, "cagr_2005_2020"),        # los que más caen
    ]).drop_duplicates(subset=["Country"])

    for _, r in to_label.iterrows():
        ax.annotate(r["Country"],
                    xy=(r["cagr_1990_2005"], r["cagr_2005_2020"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, color=COLORS["ink"], family="serif",
                    weight="bold")

    ax.set_xlabel("CAGR del consumo 1990 → 2005 (%)", fontsize=10)
    ax.set_ylabel("CAGR del consumo 2005 → 2020 (%)", fontsize=10)

    # Leyenda regional
    legend_y = 0.06
    fig.text(0.08, legend_y, "REGIÓN:", fontsize=8.5, color=COLORS["muted"],
             weight="bold", family="sans-serif")
    xpos = 0.16
    for region, color in REGION_COLORS.items():
        fig.scatter = fig.text(xpos, legend_y, "●", color=color, fontsize=12)
        fig.text(xpos+0.012, legend_y+0.002, region, fontsize=8.5,
                 color=COLORS["ink"], family="sans-serif")
        xpos += 0.13

    fig.text(0.92, legend_y, "tamaño = consumo promedio",
             fontsize=8, color=COLORS["muted"], ha="right",
             style="italic", family="sans-serif")

    footer(fig)
    out = FIG / "03_acceleration_quadrant.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 4 — EL HEATMAP TEMPORAL
# Tesis: visualización densa que muestra el "ADN" de crecimiento por país.
# Permite identificar de un vistazo patrones por región y época.
# ============================================================================
def fig4_heatmap(long, wide):
    year_cols = [c for c in wide.columns if "/" in c]

    # Normalizar cada país a su propio máximo (índice 0-100)
    # para mostrar la FORMA del crecimiento, no el nivel
    df_norm = wide.set_index("Country")[year_cols].copy()
    df_norm = df_norm.div(df_norm.max(axis=1), axis=0) * 100

    # Ordenar por región y luego por tamaño
    order_df = wide[["Country", "region", "mean_consumption"]].copy()
    region_order = ["South America", "Central America", "Caribbean", "Africa", "Asia"]
    order_df["region_rank"] = order_df["region"].apply(
        lambda r: region_order.index(r) if r in region_order else 99
    )
    order_df = order_df.sort_values(["region_rank", "mean_consumption"], ascending=[True, False])
    df_norm = df_norm.loc[order_df["Country"]]

    fig = plt.figure(figsize=(14, 10))
    title_block(
        fig,
        eyebrow="PATRÓN TEMPORAL  ·  HUELLA DE CRECIMIENTO",
        title="El ADN del crecimiento, país por país",
        subtitle="Cada fila normalizada a su propio máximo. Revela la FORMA del crecimiento (no el nivel): explosivo, gradual, estancado o regresivo.",
    )

    ax = fig.add_axes([0.22, 0.06, 0.70, 0.79])

    # Custom colormap café
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "coffee", [COLORS["bg"], "#E8C9A0", COLORS["amber"], COLORS["espresso"]]
    )

    im = ax.imshow(df_norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    # Ticks
    ax.set_xticks(range(0, len(year_cols), 3))
    ax.set_xticklabels([year_cols[i] for i in range(0, len(year_cols), 3)],
                       rotation=0, fontsize=8)
    ax.set_yticks(range(len(df_norm)))
    ax.set_yticklabels(df_norm.index, fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=2)

    # Separadores de región
    cumulative = 0
    region_breaks = []
    for region in region_order:
        n = (order_df["region"] == region).sum()
        if n > 0:
            region_breaks.append((cumulative, cumulative + n, region))
            cumulative += n
    for start, end, region in region_breaks[:-1]:
        ax.axhline(end - 0.5, color=COLORS["bg"], lw=2)

    # Etiquetas de región a la izquierda — en sus propios ejes
    label_ax = fig.add_axes([0.04, 0.06, 0.04, 0.79])
    label_ax.axis("off")
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(len(df_norm), 0)  # invertido para alinear con imshow
    for start, end, region in region_breaks:
        center = (start + end) / 2 - 0.5
        # Línea vertical
        label_ax.plot([0.9, 0.9], [start, end-0.5], color=COLORS["amber"], lw=2)
        label_ax.text(0.6, center, region.upper(),
                      fontsize=8.5, color=COLORS["amber"], weight="bold",
                      family="sans-serif", rotation=90, va="center", ha="center")

    # Colorbar
    cax = fig.add_axes([0.22, 0.025, 0.70, 0.012])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=COLORS["muted"])
    cb.set_label("% del máximo histórico de cada país", fontsize=8.5,
                 color=COLORS["muted"], labelpad=4)

    footer(fig, text="Cada fila normalizada al máximo del país (0-100%) · Fuente: ICO")
    out = FIG / "04_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 5 — LA HISTORIA REGIONAL
# Tesis: el crecimiento no es homogéneo. Asia explotó (10x), África creció
# moderado, LATAM consolidó volumen. Cada región tiene su propio playbook.
# ============================================================================
def fig5_regional_story(long):
    regional = long.groupby(["year", "region"])["consumption_cups"].sum().reset_index()

    # Normalizar a 1990 = 100 para comparar trayectorias
    base = regional[regional["year"] == 1990].set_index("region")["consumption_cups"]
    regional["index_1990"] = regional.apply(
        lambda r: r["consumption_cups"] / base[r["region"]] * 100, axis=1
    )

    fig = plt.figure(figsize=(13, 7.5))
    title_block(
        fig,
        eyebrow="REGIONES  ·  TRAYECTORIAS COMPARADAS (1990 = 100)",
        title="Asia multiplicó su consumo por 4. África lo hizo por 2.5.",
        subtitle="No existe \"el mercado\". Existen mercados regionales con dinámicas y velocidades distintas. La estrategia de exportación no puede ser una sola.",
    )

    ax = fig.add_axes([0.07, 0.16, 0.62, 0.66])

    region_order = ["Asia", "Africa", "South America", "Central America", "Caribbean"]
    for region in region_order:
        sub = regional[regional["region"] == region]
        ax.plot(sub["year"], sub["index_1990"],
                color=REGION_COLORS[region], lw=2.2, label=region, zorder=3)

    ax.axhline(100, color=COLORS["muted"], lw=0.6, ls=":", alpha=0.7)
    ax.text(1990, 102, "Base 1990 = 100", fontsize=7.5, color=COLORS["muted"], style="italic")

    # Etiquetas inline al final
    last = regional[regional["year"] == regional["year"].max()]
    for _, r in last.iterrows():
        if r["region"] in region_order:
            ax.text(r["year"] + 0.3, r["index_1990"], f"  {r['region']}",
                    fontsize=9, color=REGION_COLORS[r["region"]],
                    family="serif", weight="bold", va="center")

    ax.set_xlim(1990, 2024)
    ax.set_ylabel("Índice (1990 = 100)", fontsize=10)

    # Panel lateral: tabla de multiplicadores
    box = fig.add_axes([0.74, 0.16, 0.22, 0.66])
    box.axis("off")
    box.text(0, 1.0, "MULTIPLICADOR 30 AÑOS", fontsize=8.5, color=COLORS["amber"],
             weight="bold", family="sans-serif", transform=box.transAxes)

    summary = []
    for region in region_order:
        sub = regional[regional["region"] == region]
        mult = sub["index_1990"].iloc[-1] / 100
        summary.append((region, mult))
    summary.sort(key=lambda x: -x[1])

    for i, (region, mult) in enumerate(summary):
        y = 0.88 - i * 0.13
        box.add_patch(Rectangle((0, y-0.05), 0.025, 0.08,
                                facecolor=REGION_COLORS[region],
                                transform=box.transAxes, clip_on=False))
        box.text(0.05, y+0.005, region, fontsize=10, color=COLORS["ink"],
                 family="serif", transform=box.transAxes)
        box.text(1.0, y+0.005, f"{mult:.1f}×", fontsize=14, color=COLORS["espresso"],
                 weight="bold", family="serif", ha="right", transform=box.transAxes)

    footer(fig)
    out = FIG / "05_regional_story.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 6 — ARABICA vs ROBUSTA
# Tesis: la composición de tipos importa para política de abastecimiento.
# Los mercados que ya consumen Robusta domésticamente sugieren cosecha tipo
# Robusta y patrón distinto de precios.
# ============================================================================
def fig6_coffee_type(long):
    type_yearly = long.groupby(["year", "Coffee type"])["consumption_cups"].sum().reset_index()
    type_pivot = type_yearly.pivot(index="year", columns="Coffee type", values="consumption_cups")

    # Ordenar columnas de mayor a menor consumo promedio
    order = type_pivot.mean().sort_values(ascending=False).index.tolist()
    type_pivot = type_pivot[order]

    fig = plt.figure(figsize=(13, 7))
    title_block(
        fig,
        eyebrow="COMPOSICIÓN  ·  ARABICA Y ROBUSTA",
        title="La mezcla de tipos en el consumo doméstico cambió poco — pero importa",
        subtitle="Los países productores de Arabica/Robusta dominan absolutamente. La composición revela qué tipo de café se está \"reteniendo\" en origen.",
    )

    ax = fig.add_axes([0.07, 0.16, 0.55, 0.66])
    type_colors = {
        "Arabica/Robusta": COLORS["espresso"],
        "Robusta/Arabica": COLORS["amber"],
        "Arabica": COLORS["olive"],
        "Robusta": COLORS["terracotta"],
    }
    colors_list = [type_colors.get(c, COLORS["slate"]) for c in order]

    ax.stackplot(type_pivot.index, type_pivot.T.values,
                 labels=order, colors=colors_list,
                 alpha=0.92, edgecolor=COLORS["bg"], lw=0.4)

    ax.set_ylabel("Consumo (tazas)", fontsize=10)
    ax.set_xlim(1990, 2019)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_billions))

    # Panel derecho: share por tipo en 2020 + cambio
    box = fig.add_axes([0.68, 0.16, 0.28, 0.66])
    box.axis("off")
    box.text(0, 1.0, "PARTICIPACIÓN EN 2019/20", fontsize=8.5, color=COLORS["amber"],
             weight="bold", family="sans-serif", transform=box.transAxes)

    last_year = type_pivot.index.max()
    first_year = type_pivot.index.min()
    total_last = type_pivot.loc[last_year].sum()
    total_first = type_pivot.loc[first_year].sum()

    for i, t in enumerate(order):
        share_last = type_pivot.loc[last_year, t] / total_last * 100
        share_first = type_pivot.loc[first_year, t] / total_first * 100
        delta = share_last - share_first
        y = 0.88 - i * 0.18

        box.add_patch(Rectangle((0, y-0.06), 0.025, 0.10,
                                facecolor=type_colors.get(t, COLORS["slate"]),
                                transform=box.transAxes, clip_on=False))
        box.text(0.05, y+0.015, t, fontsize=9.5, color=COLORS["ink"],
                 family="serif", weight="bold", transform=box.transAxes)
        box.text(0.05, y-0.025, f"{share_last:.1f}% del total",
                 fontsize=8.5, color=COLORS["muted"], family="serif",
                 transform=box.transAxes)
        delta_str = f"{delta:+.1f}pp vs 1990"
        delta_color = COLORS["olive"] if delta > 0 else COLORS["deep_red"]
        box.text(0.05, y-0.055, delta_str, fontsize=8, color=delta_color,
                 family="sans-serif", weight="bold", transform=box.transAxes)

    footer(fig)
    out = FIG / "06_coffee_type.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 7 — VOLATILIDAD vs TAMAÑO
# Tesis: clasifica los mercados para forecasting. Los grandes y estables
# son la base. Los grandes y volátiles requieren modelos más sofisticados.
# ============================================================================
def fig7_volatility_size(wide):
    df = wide[wide["mean_consumption"] > 100_000].copy()

    fig = plt.figure(figsize=(13, 8))
    title_block(
        fig,
        eyebrow="VOLATILIDAD  ·  PRIORIZACIÓN DE FORECASTING",
        title="¿Qué mercados son predecibles y cuáles son una caja negra?",
        subtitle="Cuadrante superior-derecho: mercados grandes pero impredecibles. Requieren modelos cuidadosos. Inferior-derecho: maduros y estables (base confiable).",
    )

    ax = fig.add_axes([0.08, 0.13, 0.84, 0.71])

    # Líneas medianas como referencia
    median_size = df["mean_consumption"].median()
    median_cv = df["cv_consumption"].median()
    ax.axvline(median_size, color=COLORS["rule"], lw=1, ls="--", zorder=1)
    ax.axhline(median_cv, color=COLORS["rule"], lw=1, ls="--", zorder=1)

    # Etiquetas de cuadrantes
    ax.text(0.97, 0.97, "GRANDES & VOLÁTILES\nforecasting prioritario y difícil",
            transform=ax.transAxes, ha="right", va="top",
            color=COLORS["terracotta"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.03, 0.97, "PEQUEÑOS & VOLÁTILES\nbajo prioridad",
            transform=ax.transAxes, ha="left", va="top",
            color=COLORS["muted"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.97, 0.03, "GRANDES & ESTABLES\nbase confiable",
            transform=ax.transAxes, ha="right", va="bottom",
            color=COLORS["olive"], fontsize=9, weight="bold", family="sans-serif")
    ax.text(0.03, 0.03, "PEQUEÑOS & ESTABLES\nmodelos simples",
            transform=ax.transAxes, ha="left", va="bottom",
            color=COLORS["slate"], fontsize=9, weight="bold", family="sans-serif")

    colors = df["region"].map(REGION_COLORS).fillna(COLORS["muted"])
    sizes = np.log10(df["mean_consumption"]) * 25

    ax.scatter(df["mean_consumption"], df["cv_consumption"],
               s=sizes, c=colors, alpha=0.6,
               edgecolors=COLORS["ink"], linewidths=0.5, zorder=3)

    # Etiquetar países notables
    notable = pd.concat([
        df.nlargest(8, "mean_consumption"),
        df.nlargest(3, "cv_consumption"),
        df.nsmallest(3, "cv_consumption"),
    ]).drop_duplicates(subset=["Country"])
    for _, r in notable.iterrows():
        ax.annotate(r["Country"], xy=(r["mean_consumption"], r["cv_consumption"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color=COLORS["ink"], family="serif", weight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Consumo promedio (tazas, escala log)", fontsize=10)
    ax.set_ylabel("Coeficiente de variación (σ/μ)", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_billions))

    footer(fig)
    out = FIG / "07_volatility_size.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# MAIN
# ============================================================================
def main():
    apply_style()
    long, wide = load_data()
    print("Construyendo visualizaciones del Módulo 1...\n")

    outputs = []
    outputs.append(fig1_macro_story(long));        print("  ✓ 01 — Historia macro")
    outputs.append(fig2_concentration(long, wide)); print("  ✓ 02 — Concentración del mercado")
    outputs.append(fig3_acceleration_quadrant(wide)); print("  ✓ 03 — Cuadrante de aceleración")
    outputs.append(fig4_heatmap(long, wide));      print("  ✓ 04 — Heatmap temporal")
    outputs.append(fig5_regional_story(long));     print("  ✓ 05 — Historia regional")
    outputs.append(fig6_coffee_type(long));        print("  ✓ 06 — Composición Arabica/Robusta")
    outputs.append(fig7_volatility_size(wide));    print("  ✓ 07 — Volatilidad vs tamaño")

    print(f"\nGuardadas {len(outputs)} figuras en {FIG}/")
    return outputs


if __name__ == "__main__":
    main()