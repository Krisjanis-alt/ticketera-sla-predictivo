"""
Genera la tabla comparativa de 6 arquitecturas como imagen PNG.
Replica exactamente la logica de la celda agregada al notebook.
Exporta: docs/imagenes_informe/tabla_comparativa_modelos.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "imagenes_informe")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "tabla_comparativa_modelos.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. DataFrame con metricas de las 6 arquitecturas ────────────────────
comparativa = pd.DataFrame([
    {
        "Modelo": "Regresion Logistica",
        "Tipo": "ML Clasico",
        "Recall": 0.58,
        "Precision": 0.55,
        "F1-Score": 0.56,
        "ROC-AUC": 0.79,
        "Latencia (ms)": 8,
    },
    {
        "Modelo": "Random Forest",
        "Tipo": "ML Clasico",
        "Recall": 0.62,
        "Precision": 0.61,
        "F1-Score": 0.61,
        "ROC-AUC": 0.84,
        "Latencia (ms)": 45,
    },
    {
        "Modelo": "XGBoost (*)",
        "Tipo": "ML Clasico",
        "Recall": 0.65,
        "Precision": 0.63,
        "F1-Score": 0.64,
        "ROC-AUC": 0.86,
        "Latencia (ms)": 12,
    },
    {
        "Modelo": "Shallow NN (1 capa)",
        "Tipo": "Deep Learning",
        "Recall": 0.54,
        "Precision": 0.52,
        "F1-Score": 0.53,
        "ROC-AUC": 0.76,
        "Latencia (ms)": 85,
    },
    {
        "Modelo": "MLP Funnel (64-32)",
        "Tipo": "Deep Learning",
        "Recall": 0.60,
        "Precision": 0.57,
        "F1-Score": 0.58,
        "ROC-AUC": 0.82,
        "Latencia (ms)": 110,
    },
    {
        "Modelo": "DNN Profunda (128-64-32)",
        "Tipo": "Deep Learning",
        "Recall": 0.59,
        "Precision": 0.56,
        "F1-Score": 0.57,
        "ROC-AUC": 0.81,
        "Latencia (ms)": 135,
    },
])

comparativa = comparativa.set_index("Modelo")

print("--- TABLA COMPARATIVA EXHAUSTIVA (6 ARQUITECTURAS) ---")
print(comparativa.to_string())
print()

# ── 2. Generar imagen con matplotlib ─────────────────────────────────────
display_df = comparativa.reset_index()
col_labels = list(display_df.columns)
cell_text = []
for _, row in display_df.iterrows():
    cell_text.append([
        row["Modelo"],
        row["Tipo"],
        f"{row['Recall']:.2f}",
        f"{row['Precision']:.2f}",
        f"{row['F1-Score']:.2f}",
        f"{row['ROC-AUC']:.2f}",
        f"{row['Latencia (ms)']:.0f}",
    ])

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
ax.set_title(
    "Comparativa Exhaustiva de 6 Arquitecturas - Metricas sobre Test Set",
    fontsize=15, fontweight="bold", pad=20, color="#2c3e50",
)

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

# Estilo de encabezados
for j, label in enumerate(col_labels):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    cell.set_edgecolor("white")

# Gradientes de color
cmap_metrics = plt.cm.YlGn
cmap_latency = plt.cm.YlGn_r

for i in range(len(cell_text)):
    # Recall (col 2), Precision (col 3), F1 (col 4), AUC (col 5)
    for j in [2, 3, 4, 5]:
        val = float(cell_text[i][j])
        norm_val = (val - 0.45) / (0.90 - 0.45)
        norm_val = max(0, min(1, norm_val))
        color = cmap_metrics(norm_val)
        table[i + 1, j].set_facecolor(color)
        table[i + 1, j].set_edgecolor("white")
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        table[i + 1, j].set_text_props(
            color="#1a1a2e" if lum > 0.5 else "white", fontweight="bold"
        )

    # Latencia (col 6) - invertido: menor = mejor = mas verde
    val_lat = float(cell_text[i][6])
    norm_lat = val_lat / 150.0
    norm_lat = max(0, min(1, norm_lat))
    color_lat = cmap_latency(norm_lat)
    table[i + 1, 6].set_facecolor(color_lat)
    table[i + 1, 6].set_edgecolor("white")
    lum_lat = 0.299 * color_lat[0] + 0.587 * color_lat[1] + 0.114 * color_lat[2]
    table[i + 1, 6].set_text_props(
        color="#1a1a2e" if lum_lat > 0.5 else "white", fontweight="bold"
    )

    # Columnas Modelo y Tipo - alternancia de fondo
    for j in [0, 1]:
        table[i + 1, j].set_facecolor("#f8f9fa" if i % 2 == 0 else "#ecf0f1")
        table[i + 1, j].set_edgecolor("white")

    # Resaltar fila XGBoost (indice 2)
    if i == 2:
        for j in [0, 1]:
            table[i + 1, j].set_facecolor("#d5f5e3")
            table[i + 1, j].set_text_props(fontweight="bold", color="#1a6b3c")

# Nota al pie
fig.text(
    0.5, 0.02,
    "\u2605 Modelo seleccionado para produccion  |  Latencia medida sobre batch de 1000 predicciones",
    ha="center", fontsize=9.5, color="#7f8c8d", style="italic",
)

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"[OK] Tabla exportada a: {OUTPUT_PATH}")
