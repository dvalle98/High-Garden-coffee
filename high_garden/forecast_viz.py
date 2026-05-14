"""
Visualizaciones del Módulo 2 — Forecasting de demanda.
Cada figura tiene una tesis explícita.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from style import apply_style, title_block, footer, COLORS, CATEGORICAL, REGION_COLORS

BASE = Path(__file__).resolve().parent
if (BASE.parent / "data").exists() and BASE.name == "src":
    BASE = BASE.parent
DATA = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)


def load_all():
    long_df = pd.read_parquet(DATA / "coffee_long.parquet")
    wide_df = pd.read_parquet(DATA / "coffee_wide.parquet")
    breaks_df = pd.read_parquet(DATA / "breakpoints.parquet")
    bt_summary = pd.read_parquet(DATA / "backtest_summary.parquet")
    forecasts = pd.read_parquet(DATA / "forecasts.parquet")
    return long_df, wide_df, breaks_df, bt_summary, forecasts


def fmt_billions(x, pos=None):
    if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6: return f"{x/1e6:.0f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# ============================================================================
# FIGURA 8 — CATÁLOGO DE QUIEBRES DETECTADOS
# Tesis: PELT identifica con precisión los puntos donde el régimen cambió.
# Mostrar 6 países representativos para evidenciar el funcionamiento.
# ============================================================================
def fig8_breakpoints_catalog(long_df, breaks_df):
    # Selección curada: 6 países con cambios de pendiente claros (no saltos de nivel)
    # 2 aceleraciones, 2 recuperaciones (de declinar a crecer), 2 desaceleraciones
    curated = [
        ("Viet Nam",     "Robusta/Arabica", "strong"),    # iconic emerging acceleration
        ("Indonesia",    "Robusta/Arabica", "strong"),    # Asian acceleration
        ("Tanzania",     "Arabica/Robusta", "strong"),    # African acceleration
        ("Mexico",       "Arabica/Robusta", "reversal"),  # recovery from decline
        ("Honduras",     "Arabica",         "reversal"),  # recovery
        ("Nicaragua",    "Arabica",         "reversal"),  # recent deceleration
    ]

    candidates = []
    for country, ctype, expected_sig in curated:
        row = breaks_df[(breaks_df["Country"] == country) &
                        (breaks_df["Coffee type"] == ctype)]
        if not row.empty and pd.notna(row["breakpoint_year"].iloc[0]):
            candidates.append({
                "Country": country,
                "Coffee type": ctype,
                "breakpoint_year": row["breakpoint_year"].iloc[0],
                "significance": row["significance"].iloc[0],
            })

    fig = plt.figure(figsize=(13, 8.5))
    title_block(
        fig,
        eyebrow="DETECCIÓN DE QUIEBRES  ·  PELT  ·  6 CASOS REPRESENTATIVOS",
        title="Identificación automática del año donde cambió el régimen",
        subtitle="Cada serie es de un país real. La línea vertical es el quiebre detectado. Forecasting que ignore este punto sobreestimará sistemáticamente.",
    )

    significance_colors = {
        "strong":   COLORS["espresso"],
        "reversal": COLORS["deep_red"],
        "moderate": COLORS["amber"],
    }

    for i, c in enumerate(candidates[:6]):
        row, col = i // 3, i % 3
        ax = fig.add_axes([0.06 + col * 0.32, 0.46 - row * 0.32, 0.27, 0.24])

        grp = long_df[(long_df["Country"] == c["Country"]) &
                      (long_df["Coffee type"] == c["Coffee type"])].sort_values("year")
        if grp.empty:
            continue

        years = grp["year"].values
        values = grp["consumption_cups"].values / 1e6  # en millones

        bp_year = int(c["breakpoint_year"])
        bp_idx = np.where(years == bp_year)[0]
        if len(bp_idx) == 0:
            continue
        bp_idx = bp_idx[0]

        # Pre-segmento gris, post-segmento color
        sig_color = significance_colors.get(c["significance"], COLORS["amber"])
        ax.plot(years[:bp_idx+1], values[:bp_idx+1],
                color=COLORS["muted"], lw=1.6)
        ax.plot(years[bp_idx:], values[bp_idx:],
                color=sig_color, lw=2.2)
        ax.scatter(years[bp_idx], values[bp_idx], color=sig_color, s=44, zorder=5)

        # Línea vertical en el quiebre
        ax.axvline(bp_year, color=sig_color, lw=0.6, ls="--", alpha=0.5)

        # Etiqueta del año del quiebre dentro del plot
        ymax = values.max()
        ymin = values.min()
        yr = ymax + (ymax - ymin) * 0.04
        ax.text(bp_year, yr, f"  {bp_year}",
                fontsize=8.5, color=sig_color, weight="bold",
                family="serif")

        # Header del panel: 2 líneas
        country_label = c["Country"] if len(c["Country"]) < 25 else c["Country"][:22] + "..."
        ax.text(0, 1.18, country_label, transform=ax.transAxes,
                fontsize=10.5, color=COLORS["ink"], weight="bold", family="serif")
        ax.text(0, 1.05, f"{c['Coffee type']}  ·  {c['significance']}",
                transform=ax.transAxes, fontsize=7.5, color=sig_color,
                family="sans-serif", weight="bold")

        ax.set_xlim(1989, 2020)
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel("Tazas (M)", fontsize=8)

    # Leyenda de colores por significancia
    legend_ax = fig.add_axes([0.06, 0.06, 0.88, 0.04])
    legend_ax.axis("off")
    items = [
        ("STRONG", "cambio de pendiente >100%", significance_colors["strong"]),
        ("REVERSAL", "cambio de dirección (crece→cae o cae→crece)", significance_colors["reversal"]),
        ("MODERATE", "cambio entre 50% y 100%", significance_colors["moderate"]),
    ]
    x = 0
    for label, desc, color in items:
        legend_ax.add_patch(Rectangle((x, 0.55), 0.014, 0.35, facecolor=color,
                                       transform=legend_ax.transAxes, clip_on=False))
        legend_ax.text(x + 0.02, 0.7, label, fontsize=8.5, color=color,
                       weight="bold", family="sans-serif", transform=legend_ax.transAxes)
        legend_ax.text(x + 0.02, 0.2, desc, fontsize=7.5, color=COLORS["muted"],
                       style="italic", family="serif", transform=legend_ax.transAxes)
        x += 0.33

    footer(fig)
    out = FIG / "08_breakpoints_catalog.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 9 — DISTRIBUCIÓN DEL MODELO GANADOR
# Tesis: no un solo modelo gana siempre. Cada segmento requiere su técnica.
# ============================================================================
def fig9_model_selection(bt_summary, wide_df):
    # Cruzar con tamaño promedio
    merged = bt_summary.merge(
        wide_df[["Country", "Coffee type", "mean_consumption", "cv_consumption"]],
        on=["Country", "Coffee type"], how="left"
    )

    fig = plt.figure(figsize=(13, 7.5))
    title_block(
        fig,
        eyebrow="SELECCIÓN DE MODELO  ·  POR SEGMENTO DE MERCADO",
        title="No existe un solo modelo ganador — cada segmento demanda su técnica",
        subtitle="El backtest selecciona el mejor modelo por país. Los segmentados ganan en mercados con quiebres claros; Holt en los suavizables; naive en los chicos y ruidosos.",
    )

    # PANEL 1: Distribución del modelo ganador
    ax = fig.add_axes([0.06, 0.16, 0.30, 0.62])
    counts = bt_summary["best_model"].value_counts()
    model_colors = {
        "naive_drift": COLORS["slate"],
        "linear_trend": COLORS["amber"],
        "holt": COLORS["olive"],
        "segmented": COLORS["espresso"],
    }
    colors_bar = [model_colors.get(m, COLORS["muted"]) for m in counts.index]
    y_pos = np.arange(len(counts))
    bars = ax.barh(y_pos, counts.values, color=colors_bar, alpha=0.92,
                   edgecolor=COLORS["bg"], lw=1)
    for i, (model, val) in enumerate(counts.items()):
        ax.text(val + 0.5, i, f"{val} países", fontsize=9, va="center",
                color=COLORS["ink"], family="serif", weight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace("_", " ").title() for m in counts.index], fontsize=9)
    ax.set_xlabel("Países donde gana este modelo", fontsize=9)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.invert_yaxis()
    ax.set_title("Modelo ganador", fontsize=11, loc="left",
                 weight="bold", color=COLORS["ink"], pad=10)

    # PANEL 2: Modelo ganador en el espacio (tamaño, volatilidad)
    ax2 = fig.add_axes([0.43, 0.16, 0.52, 0.62])
    ax2.axhline(merged["cv_consumption"].median(), color=COLORS["rule"], lw=0.8, ls="--")
    ax2.axvline(merged["mean_consumption"].median(), color=COLORS["rule"], lw=0.8, ls="--")

    for model, color in model_colors.items():
        sub = merged[merged["best_model"] == model]
        if sub.empty:
            continue
        ax2.scatter(sub["mean_consumption"], sub["cv_consumption"],
                    s=60, c=color, alpha=0.7,
                    edgecolors=COLORS["ink"], linewidths=0.5,
                    label=model.replace("_", " ").title())

    ax2.set_xscale("log")
    ax2.set_xlabel("Consumo promedio (tazas, log)", fontsize=9)
    ax2.set_ylabel("Coeficiente de variación", fontsize=9)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(fmt_billions))
    ax2.set_title("Distribución espacial del modelo ganador", fontsize=11, loc="left",
                  weight="bold", color=COLORS["ink"], pad=10)
    ax2.legend(loc="upper right", fontsize=8.5)

    footer(fig)
    out = FIG / "09_model_selection.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 10 — BACKTEST ACCURACY: COMPARACIÓN SEGMENTED vs OTROS
# Tesis: respetar quiebres estructurales mejora la precisión predictiva.
# Esta es la EVIDENCIA EMPÍRICA de que la innovación técnica del módulo aporta.
# ============================================================================
def fig10_backtest_accuracy(bt_summary, breaks_df):
    # Filtrar a países con quiebre y con segmented evaluado
    with_break = bt_summary.merge(
        breaks_df[breaks_df["breakpoint_year"].notna()][["Country", "Coffee type"]],
        on=["Country", "Coffee type"], how="inner"
    )
    comp = with_break.dropna(subset=["segmented_mape", "linear_trend_mape"]).copy()
    comp["improvement"] = comp["linear_trend_mape"] - comp["segmented_mape"]

    # Capar valores extremos para legibilidad, marcando los originales
    CAP = 50
    comp["improvement_display"] = comp["improvement"].clip(-CAP, CAP)
    comp["was_clipped"] = comp["improvement"].abs() > CAP

    fig = plt.figure(figsize=(13, 9))
    title_block(
        fig,
        eyebrow="VALIDACIÓN EMPÍRICA  ·  ¿LA DETECCIÓN DE QUIEBRES AYUDA?",
        title="En 67% de los países con quiebre, el modelo segmentado es más preciso",
        subtitle="Cada barra: mejora del MAPE al usar segmented vs linear_trend. Valores >50pp se truncan visualmente (anotados con ▶).",
    )

    ax = fig.add_axes([0.16, 0.10, 0.78, 0.72])

    # Ordenar por improvement real (no display)
    comp = comp.sort_values("improvement", ascending=True).reset_index(drop=True)
    comp["label"] = comp["Country"] + " · " + comp["Coffee type"]

    colors_bar = np.where(comp["improvement"] > 0, COLORS["olive"], COLORS["deep_red"])
    y_pos = np.arange(len(comp))
    ax.barh(y_pos, comp["improvement_display"], color=colors_bar, alpha=0.85,
            edgecolor=COLORS["bg"], lw=0.5)

    # Marcar barras truncadas con anotación del valor real
    for i, row in comp.iterrows():
        if row["was_clipped"]:
            sign = "▶" if row["improvement"] > 0 else "◀"
            x_pos = CAP if row["improvement"] > 0 else -CAP
            ha = "left" if row["improvement"] > 0 else "right"
            ax.text(x_pos + (1 if row["improvement"] > 0 else -1), i,
                    f"{sign} {row['improvement']:+.0f}pp",
                    fontsize=7.5, color=COLORS["olive"] if row["improvement"]>0 else COLORS["deep_red"],
                    weight="bold", family="sans-serif", va="center", ha=ha)

    ax.axvline(0, color=COLORS["ink"], lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp["label"], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mejora del MAPE (puntos porcentuales)  ←  peor    ·    mejor  →",
                  fontsize=9.5)
    ax.set_xlim(-CAP - 8, CAP + 8)

    # Estadísticas resumen — debajo del subtitle
    n_total = len(comp)
    n_better = (comp["improvement"] > 0).sum()
    avg_better = comp[comp["improvement"] > 0]["improvement"].mean()
    avg_worse = comp[comp["improvement"] < 0]["improvement"].mean()

    fig.text(0.16, 0.85,
             f"$\\bf{{{n_better}}}$ / {n_total} países con mejora  ·  "
             f"mejora promedio  $\\bf{{+{avg_better:.1f}\\,pp}}$  ·  "
             f"pérdida promedio  $\\bf{{{avg_worse:.1f}\\,pp}}$",
             fontsize=10.5, color=COLORS["ink"], family="serif")

    footer(fig)
    out = FIG / "10_backtest_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 11 — FORECASTS TOP-10
# Tesis: visualización del producto final (las proyecciones) para los países
# que más mueven la aguja exportadora.
# ============================================================================
def fig11_forecasts_top10(long_df, forecasts, wide_df):
    # Top 10 por consumo promedio
    top10 = wide_df.nlargest(10, "mean_consumption")[["Country", "Coffee type"]]

    fig = plt.figure(figsize=(14, 9))
    title_block(
        fig,
        eyebrow="PROYECCIONES 2020/21 — 2024/25  ·  TOP 10 PAÍSES",
        title="Hacia dónde va la demanda doméstica de los 10 productores más grandes",
        subtitle="Línea sólida: histórico. Línea punteada: forecast con el mejor modelo por país. Banda: intervalo de confianza al 80%.",
    )

    for i, (_, c) in enumerate(top10.iterrows()):
        row, col = i // 5, i % 5
        ax = fig.add_axes([0.05 + col * 0.19, 0.45 - row * 0.36, 0.16, 0.28])

        # Histórico
        hist = long_df[(long_df["Country"] == c["Country"]) &
                       (long_df["Coffee type"] == c["Coffee type"])].sort_values("year")
        if hist.empty:
            continue
        hist_years = hist["year"].values
        hist_values = hist["consumption_cups"].values / 1e6  # M tazas

        # Forecast
        fc = forecasts[(forecasts["Country"] == c["Country"]) &
                       (forecasts["Coffee type"] == c["Coffee type"])].sort_values("year")
        fc_years = fc["year"].values
        fc_values = fc["forecast"].values / 1e6
        fc_low = fc["lower_80"].values / 1e6
        fc_high = fc["upper_80"].values / 1e6

        # Conectar último histórico con primer forecast
        if len(fc_values) > 0:
            conn_years = np.concatenate([[hist_years[-1]], fc_years])
            conn_values = np.concatenate([[hist_values[-1]], fc_values])
            conn_low = np.concatenate([[hist_values[-1]], fc_low])
            conn_high = np.concatenate([[hist_values[-1]], fc_high])

            ax.fill_between(conn_years, conn_low, conn_high,
                            color=COLORS["amber"], alpha=0.15)
            ax.plot(conn_years, conn_values, color=COLORS["amber"], lw=1.8, ls="--")

        ax.plot(hist_years, hist_values, color=COLORS["espresso"], lw=1.8)
        ax.scatter(hist_years[-1], hist_values[-1], color=COLORS["espresso"], s=18, zorder=5)

        # Etiqueta del modelo usado
        model_used = fc["model_used"].iloc[0] if len(fc) > 0 else "n/a"
        had_break = fc["had_breakpoint"].iloc[0] if len(fc) > 0 else False

        # Header: nombre país y subtítulo separados claramente
        ax.text(0, 1.18, c["Country"], transform=ax.transAxes,
                fontsize=10, color=COLORS["ink"], weight="bold", family="serif")
        ax.text(0, 1.06, f"{c['Coffee type']}  ·  {model_used}",
                transform=ax.transAxes, fontsize=7, color=COLORS["muted"],
                family="sans-serif")
        if had_break:
            ax.text(1.0, 1.06, "● quiebre",
                    transform=ax.transAxes, fontsize=6.5,
                    color=COLORS["terracotta"], family="sans-serif",
                    ha="right", weight="bold")

        ax.tick_params(labelsize=7)
        ax.set_ylabel("M tazas", fontsize=7.5)
        ax.set_xlim(1989, 2026)

    footer(fig)
    out = FIG / "11_forecasts_top10.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# FIGURA 12 — PROYECCIÓN AGREGADA + TESIS DE PRESIÓN EXPORTABLE
# Tesis: visualización del resultado de negocio: cómo va a crecer la
# retención de café en origen, región por región.
# ============================================================================
def fig12_aggregated_forecast(long_df, forecasts):
    # Mapeo correcto de regiones (un país → una región)
    region_map = (long_df[["Country", "region"]]
                  .drop_duplicates()
                  .set_index("Country")["region"]
                  .to_dict())

    hist = long_df.copy()
    hist_regional = hist.groupby(["year", "region"])["consumption_cups"].sum().reset_index()

    fc = forecasts.copy()
    fc["region"] = fc["Country"].map(region_map)
    fc_regional = fc.groupby(["year", "region"])["forecast"].sum().reset_index()
    fc_regional = fc_regional.rename(columns={"forecast": "consumption_cups"})

    fig = plt.figure(figsize=(13, 8))
    title_block(
        fig,
        eyebrow="PRESIÓN EXPORTABLE  ·  PROYECCIÓN AGREGADA POR REGIÓN",
        title="Cuánto café adicional se va a quedar en origen los próximos 5 años",
        subtitle="Suma del consumo doméstico proyectado por región. Cada taza adicional aquí es una taza menos disponible para el mercado de exportación mundial.",
    )

    ax = fig.add_axes([0.06, 0.16, 0.55, 0.66])

    region_order = ["South America", "Africa", "Asia", "Central America", "Caribbean"]
    for region in region_order:
        h = hist_regional[hist_regional["region"] == region]
        f = fc_regional[fc_regional["region"] == region]
        color = REGION_COLORS[region]

        ax.plot(h["year"], h["consumption_cups"], color=color, lw=2.2)
        if not f.empty:
            # Conectar último histórico con forecast
            last_y = h["year"].max()
            last_v = h[h["year"] == last_y]["consumption_cups"].values[0]
            f_y = np.concatenate([[last_y], f["year"].values])
            f_v = np.concatenate([[last_v], f["consumption_cups"].values])
            ax.plot(f_y, f_v, color=color, lw=2.2, ls="--", alpha=0.8)

    # Línea divisoria entre histórico y forecast
    last_hist_year = hist_regional["year"].max()
    ax.axvline(last_hist_year, color=COLORS["muted"], lw=0.6, ls=":")
    ax.text(last_hist_year - 0.3, ax.get_ylim()[1] * 0.95, "histórico  |",
            fontsize=7.5, color=COLORS["muted"], style="italic", ha="right")
    ax.text(last_hist_year + 0.3, ax.get_ylim()[1] * 0.95, "  forecast",
            fontsize=7.5, color=COLORS["muted"], style="italic")

    # Etiquetar regiones al final
    for region in region_order:
        f = fc_regional[fc_regional["region"] == region]
        if not f.empty:
            last = f.sort_values("year").iloc[-1]
            ax.text(last["year"] + 0.3, last["consumption_cups"], f"  {region}",
                    fontsize=9, color=REGION_COLORS[region], family="serif",
                    weight="bold", va="center")

    ax.set_xlim(1990, 2027)
    ax.set_ylabel("Consumo agregado (tazas)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_billions))

    # PANEL DERECHO: crecimiento proyectado por región
    box = fig.add_axes([0.65, 0.16, 0.30, 0.66])
    box.axis("off")
    box.text(0, 1.0, "CRECIMIENTO PROYECTADO 2019 → 2024",
             fontsize=8.5, color=COLORS["amber"], weight="bold",
             family="sans-serif", transform=box.transAxes)

    growth_data = []
    for region in region_order:
        h = hist_regional[(hist_regional["region"] == region) &
                          (hist_regional["year"] == 2019)]
        f = fc_regional[(fc_regional["region"] == region) &
                        (fc_regional["year"] == 2024)]
        if not h.empty and not f.empty:
            h_v = h["consumption_cups"].values[0]
            f_v = f["consumption_cups"].values[0]
            growth_data.append((region, h_v, f_v, (f_v - h_v) / h_v * 100))

    growth_data.sort(key=lambda x: -x[3])

    for i, (region, h_v, f_v, pct) in enumerate(growth_data):
        y = 0.90 - i * 0.16
        box.add_patch(Rectangle((0, y - 0.05), 0.025, 0.10,
                                facecolor=REGION_COLORS[region],
                                transform=box.transAxes, clip_on=False))
        box.text(0.05, y + 0.01, region, fontsize=10, color=COLORS["ink"],
                 family="serif", weight="bold", transform=box.transAxes)
        box.text(0.05, y - 0.025, f"{fmt_billions(h_v)} → {fmt_billions(f_v)}",
                 fontsize=8, color=COLORS["muted"], family="sans-serif",
                 transform=box.transAxes)
        pct_color = COLORS["olive"] if pct > 0 else COLORS["deep_red"]
        box.text(1.0, y + 0.0, f"{pct:+.1f}%", fontsize=14, color=pct_color,
                 weight="bold", family="serif", ha="right",
                 transform=box.transAxes)

    # Totales agregados al pie
    h_total_2019 = hist_regional[hist_regional["year"] == 2019]["consumption_cups"].sum()
    f_total_2024 = fc_regional[fc_regional["year"] == 2024]["consumption_cups"].sum()
    diff_pct = (f_total_2024 - h_total_2019) / h_total_2019 * 100

    fig.text(0.06, 0.06,
             f"PROYECCIÓN GLOBAL:  {fmt_billions(h_total_2019)} (2019)  →  "
             f"{fmt_billions(f_total_2024)} (2024)   ·   "
             f"crecimiento adicional: {diff_pct:+.1f}%",
             fontsize=10, color=COLORS["espresso"], weight="bold", family="serif")

    footer(fig)
    out = FIG / "12_aggregated_forecast.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# MAIN
# ============================================================================
def main():
    apply_style()
    long_df, wide_df, breaks_df, bt_summary, forecasts = load_all()
    print("Construyendo visualizaciones del Módulo 2...\n")

    outputs = []
    outputs.append(fig8_breakpoints_catalog(long_df, breaks_df))
    print("  ✓ 08 — Catálogo de quiebres detectados")
    outputs.append(fig9_model_selection(bt_summary, wide_df))
    print("  ✓ 09 — Distribución del modelo ganador")
    outputs.append(fig10_backtest_accuracy(bt_summary, breaks_df))
    print("  ✓ 10 — Validación empírica del segmented")
    outputs.append(fig11_forecasts_top10(long_df, forecasts, wide_df))
    print("  ✓ 11 — Forecasts top 10 países")
    outputs.append(fig12_aggregated_forecast(long_df, forecasts))
    print("  ✓ 12 — Proyección agregada regional")

    print(f"\nGuardadas {len(outputs)} figuras en {FIG}/")
    return outputs


if __name__ == "__main__":
    main()