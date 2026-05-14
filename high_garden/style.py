"""
Visual style module — High Garden Coffee
Define paleta, tipografía y helpers de anotación para gráficas editoriales.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# ---------- PALETA DE COLORES ----------
# Inspirada en el café: tonos cálidos, profundos, con acentos editoriales

COLORS = {
    "bg":          "#FAF7F2",   # papel envejecido
    "ink":         "#1C1410",   # tinta principal
    "muted":       "#6B5A4E",   # texto secundario
    "rule":        "#D4C9B8",   # líneas de regla
    "grid":        "#E8DFD2",   # grid muy sutil

    # Acentos principales
    "espresso":    "#3D2817",   # café espresso (color primario)
    "amber":       "#C7873B",   # ámbar (acento cálido)
    "terracotta":  "#A0492A",   # terracota (alerta cálida)
    "olive":       "#6B7A3F",   # oliva (positivo)
    "deep_red":    "#7A2E2A",   # rojo profundo (negativo fuerte)
    "teal":        "#2C5F5D",   # teal (frío informativo)
    "gold":        "#B8860B",   # oro (highlight)
    "slate":       "#475569",   # gris azulado
}

# Paleta categórica (para series múltiples)
CATEGORICAL = [
    COLORS["espresso"],
    COLORS["amber"],
    COLORS["olive"],
    COLORS["terracotta"],
    COLORS["teal"],
    COLORS["gold"],
    COLORS["deep_red"],
    COLORS["slate"],
]

# Paleta para regiones
REGION_COLORS = {
    "South America":   COLORS["espresso"],
    "Central America": COLORS["amber"],
    "Caribbean":       COLORS["gold"],
    "Africa":          COLORS["terracotta"],
    "Asia":            COLORS["teal"],
}


def apply_style():
    """Aplica el estilo editorial a matplotlib."""
    mpl.rcParams.update({
        # Figura
        "figure.facecolor": COLORS["bg"],
        "figure.dpi": 110,
        "savefig.facecolor": COLORS["bg"],
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,

        # Ejes
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["rule"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": COLORS["ink"],
        "axes.titlecolor": COLORS["ink"],
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.prop_cycle": cycler(color=CATEGORICAL),

        # Ticks
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",

        # Leyenda
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,

        # Fuentes
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "serif"],
        "font.size": 10,
    })


def title_block(fig, eyebrow, title, subtitle, x=0.06, y_top=0.96):
    """
    Bloque de título editorial:
    - eyebrow: kicker pequeño en mayúsculas
    - title: titular grande
    - subtitle: tesis del gráfico (la idea de negocio)
    """
    # Espaciado de letras simulado con espacios
    eyebrow_spaced = "  ".join(eyebrow)
    fig.text(x, y_top,        eyebrow_spaced,  fontsize=8.5, color=COLORS["amber"],
             weight="bold", family="sans-serif")
    fig.text(x, y_top - 0.04, title,    fontsize=17, color=COLORS["ink"],
             weight="bold", family="serif")
    fig.text(x, y_top - 0.075, subtitle, fontsize=10.5, color=COLORS["muted"],
             family="serif", style="italic")


def footer(fig, text="Fuente: International Coffee Organization (ICO) — Domestic consumption in exporting countries  |  High Garden Coffee — análisis técnico"):
    fig.text(0.06, 0.015, text, fontsize=7.5, color=COLORS["muted"],
             family="sans-serif")


def annotate(ax, x, y, text, **kwargs):
    """Anotación con estilo editorial."""
    defaults = dict(
        fontsize=9, color=COLORS["ink"], family="serif",
        arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.6),
    )
    defaults.update(kwargs)
    ax.annotate(text, xy=(x, y), **defaults)