"""
Reproduce el entrenamiento del MLP (Celda 13 del notebook) para capturar
el objeto `history` de Keras. Genera la curva de convergencia
(Train Loss vs Validation Loss) con la linea de Early Stopping.

Exporta: docs/imagenes_informe/curvas_convergencia.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "dataset_tickets_con_senal.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "imagenes_informe")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "curvas_convergencia.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Cargar datos (replica exacta del notebook) ───────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv(DATA_PATH)

X = df.drop(["ticket_id", "incumple_sla"], axis=1)
y = df["incumple_sla"]

# Particion Train(60%) / Val(20%) / Test(20%) -- misma semilla del notebook
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Pipeline de preprocesamiento
num_features = ["hora_creacion", "tiempo_resolucion_hrs"]
cat_features = ["dia_semana", "categoria", "prioridad", "seniority_agente"]

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features),
])

X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)

# SMOTE (igual que el notebook)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)

# ── 2. Definir y entrenar el MLP (Celda 13 del notebook) ────────────────
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

input_dim = X_train_smote.shape[1]

from tensorflow.keras.layers import Input

mlp_model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])
mlp_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

print("Entrenando MLP (epochs=100, patience=10)...")
history = mlp_model.fit(
    X_train_smote, y_train_smote,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_prep, y_val),
    callbacks=[early_stopping],
    verbose=0,
)

# ── 3. Extraer metricas del history ─────────────────────────────────────
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_ran = len(train_loss)

# Early stopping: la epoca con mejor val_loss
best_epoch = int(np.argmin(val_loss))                 # 0-indexed
stopped_epoch = epochs_ran                            # total de epocas ejecutadas
best_val_loss = val_loss[best_epoch]
best_train_loss = train_loss[best_epoch]

print(f"   Epocas ejecutadas : {stopped_epoch} / 100")
print(f"   Mejor epoca       : {best_epoch + 1}")
print(f"   Train Loss (mejor): {best_train_loss:.4f}")
print(f"   Val Loss   (mejor): {best_val_loss:.4f}")

# ── 4. Graficar curvas de convergencia ───────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)

fig, ax = plt.subplots(figsize=(10, 5.5))

epoch_range = np.arange(1, epochs_ran + 1)

# Curvas de loss
ax.plot(epoch_range, train_loss, linewidth=2.2, color="#3498db",
        label="Train Loss", marker="o", markersize=3.5, alpha=0.85)
ax.plot(epoch_range, val_loss, linewidth=2.2, color="#e74c3c",
        label="Validation Loss", marker="s", markersize=3.5, alpha=0.85)

# Linea vertical de Early Stopping (mejor epoca)
ax.axvline(
    x=best_epoch + 1,
    color="#2ecc71",
    linestyle="--",
    linewidth=2,
    label=f"Early Stopping (epoca {best_epoch + 1})",
)

# Punto destacado en la mejor epoca
ax.scatter(best_epoch + 1, best_val_loss, s=120, color="#e74c3c",
           zorder=5, edgecolors="white", linewidths=1.5)
ax.scatter(best_epoch + 1, best_train_loss, s=120, color="#3498db",
           zorder=5, edgecolors="white", linewidths=1.5)

# Anotacion en la mejor epoca
offset_y = (max(max(train_loss), max(val_loss)) - min(min(train_loss), min(val_loss))) * 0.08
ax.annotate(
    f"Best val_loss = {best_val_loss:.4f}",
    xy=(best_epoch + 1, best_val_loss),
    xytext=(best_epoch + 4, best_val_loss + offset_y),
    fontsize=10,
    fontweight="bold",
    color="#c0392b",
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
)

# Titulos y ejes
ax.set_title(
    "Curvas de Convergencia del MLP - Early Stopping",
    fontsize=16, fontweight="bold", pad=18, color="#2c3e50",
)
ax.set_xlabel("Epoca", fontsize=13, labelpad=10)
ax.set_ylabel("Loss (Binary Crossentropy)", fontsize=13, labelpad=10)
ax.legend(fontsize=11, loc="upper right", framealpha=0.9)

# Nota al pie
fig.text(
    0.5, -0.02,
    f"Entrenamiento detenido en la epoca {stopped_epoch}  |  "
    f"Pesos restaurados de la epoca {best_epoch + 1}  |  patience = 10",
    ha="center", fontsize=9.5, color="#7f8c8d", style="italic",
)

sns.despine(left=True)
plt.tight_layout()

# ── 5. Guardar ───────────────────────────────────────────────────────────
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\n[OK] Grafico guardado en: {OUTPUT_PATH}")
