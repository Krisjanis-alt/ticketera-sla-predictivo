"""
Genera un gráfico de barras (countplot) sobre la variable incumple_sla
del dataset de tickets con señal, anotando las proporciones 85%/15%.
Guarda la figura en docs/imagenes_informe/balance_clases_nuevo.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Rutas ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "dataset_tickets_con_senal.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "imagenes_informe")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "balance_clases_nuevo.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Lectura ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
total = len(df)

# Mapear 0/1 a etiquetas legibles
df["Incumple_SLA"] = df["incumple_sla"].map({0: "Cumple SLA", 1: "Incumple SLA"})

# Conteos y porcentajes
conteos = df["Incumple_SLA"].value_counts()
porcentajes = {"Cumple SLA": 85, "Incumple SLA": 15}  # Proporciones solicitadas

# ── Estilo del gráfico ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
palette = {"Cumple SLA": "#2ecc71", "Incumple SLA": "#e74c3c"}

fig, ax = plt.subplots(figsize=(8, 5.5))

# Countplot
sns.countplot(
    data=df,
    x="Incumple_SLA",
    hue="Incumple_SLA",
    order=["Cumple SLA", "Incumple SLA"],
    hue_order=["Cumple SLA", "Incumple SLA"],
    palette=palette,
    edgecolor="white",
    linewidth=1.2,
    legend=False,
    ax=ax,
)

# ── Anotaciones sobre las barras ─────────────────────────────────────────
for bar, label in zip(ax.patches, ["Cumple SLA", "Incumple SLA"]):
    height = bar.get_height()
    pct = porcentajes[label]
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + total * 0.012,
        f"{int(height):,} tickets\n({pct}%)",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#2c3e50",
    )

# ── Títulos y ejes ───────────────────────────────────────────────────────
ax.set_title(
    "Distribución de la variable Incumple_SLA",
    fontsize=16,
    fontweight="bold",
    pad=18,
    color="#2c3e50",
)
ax.set_xlabel("Estado del SLA", fontsize=13, labelpad=10)
ax.set_ylabel("Cantidad de tickets", fontsize=13, labelpad=10)
ax.set_ylim(0, conteos.max() * 1.22)

# Línea divisoria visual del porcentaje
ax.axhline(y=total * 0.5, color="#bdc3c7", linestyle="--", linewidth=0.8)

sns.despine(left=True)
plt.tight_layout()

# ── Guardar ──────────────────────────────────────────────────────────────
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"[OK] Grafico guardado en: {OUTPUT_PATH}")
print(f"   Total tickets : {total:,}")
print(f"   Cumple SLA    : {conteos['Cumple SLA']:,}  -> 85%")
print(f"   Incumple SLA  : {conteos['Incumple SLA']:,}  -> 15%")
