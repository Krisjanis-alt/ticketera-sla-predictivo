"""
Pipeline de extremo a extremo para entrenamiento del modelo XGBoost
de prediccion de incumplimiento de SLA.

Flujo:
  1. Carga el dataset desde datasets/dataset_tickets_con_senal.csv
  2. Elimina 'tiempo_resolucion_hrs' (evento futuro — anti target leakage)
  3. Particiona en Train (60%) / Val (20%) / Test (20%) con random_state=42
  4. Preprocesa (imputacion + escalado + one-hot encoding)
  5. Aplica SMOTE para balanceo + scale_pos_weight para maximizar Recall
  6. Entrena XGBClassifier con pesos de clase
  7. Imprime el classification report sobre el conjunto de Test
  8. Exporta modelo_xgboost.pkl y preprocessor.pkl hacia backend/

Modelos evaluados en el notebook complementario:
  Regresion Logistica | Random Forest | XGBoost | MLP Deep (128-64-32)

Uso:
  py run_ml.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# Rutas
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "dataset_tickets_con_senal.csv")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODEL_PATH = os.path.join(BACKEND_DIR, "modelo_xgboost.pkl")
PREPROC_PATH = os.path.join(BACKEND_DIR, "preprocessor.pkl")

RANDOM_STATE = 42


def main():
    t0 = time.time()

    # =================================================================
    # 1. CARGA DE DATOS
    # =================================================================
    print("=" * 60)
    print("  PIPELINE DE ENTRENAMIENTO - XGBoost SLA Predictivo")
    print("=" * 60)

    if not os.path.isfile(DATA_PATH):
        print(f"\n[ERROR] No se encontro el dataset en: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"\n[1/6] Dataset cargado: {DATA_PATH}")
    print(f"      Filas: {len(df):,}  |  Columnas: {df.shape[1]}")
    print(f"      Distribucion target (incumple_sla):")
    dist = df["incumple_sla"].value_counts()
    for v, c in dist.items():
        pct = c / len(df) * 100
        label = "Cumple SLA" if v == 0 else "Incumple SLA"
        print(f"        {v} ({label}): {c:,} ({pct:.1f}%)")

    # =================================================================
    # 2. ELIMINACION DE TARGET LEAKAGE
    # =================================================================
    # 'tiempo_resolucion_hrs' es el tiempo real de resolucion del ticket.
    # Es un EVENTO FUTURO que NO esta disponible al momento de la prediccion.
    # Se elimina ANTES de cualquier particion o preprocesamiento.
    df = df.drop(columns=["tiempo_resolucion_hrs"])
    print(f"\n      -> 'tiempo_resolucion_hrs' eliminada (anti-leakage). "
          f"Columnas restantes: {df.shape[1]}")

    # =================================================================
    # 3. PARTICION DE DATOS
    # =================================================================
    X = df.drop(["ticket_id", "incumple_sla"], axis=1)
    y = df["incumple_sla"]

    # Train 60% / Val 20% / Test 20%
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"\n[3/6] Particion de datos (random_state={RANDOM_STATE}):")
    print(f"      Train : {len(X_train):,} muestras (60%)")
    print(f"      Val   : {len(X_val):,} muestras (20%)")
    print(f"      Test  : {len(X_test):,} muestras (20%)")

    # =================================================================
    # 4. PREPROCESAMIENTO
    # =================================================================
    # Nota: 'hora_creacion' es la unica feature numerica.
    # 'tiempo_resolucion_hrs' fue eliminada en el paso anterior (anti-leakage).
    num_features = ["hora_creacion"]
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
    X_test_prep = preprocessor.transform(X_test)

    n_features = X_train_prep.shape[1]
    print(f"\n[4/6] Preprocesamiento completado:")
    print(f"      Features numericas: {num_features}")
    print(f"      Features categoricas: {len(cat_features)}")
    print(f"      Dimensiones post-encoding: {n_features}")

    # =================================================================
    # 5. BALANCEO CON SMOTE + CALCULO DE PESOS DE CLASE
    # =================================================================
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)

    # Calcular scale_pos_weight para maximizar Recall
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos

    print(f"\n[5/6] Balanceo de clases:")
    print(f"      SMOTE: {X_train_prep.shape[0]:,} -> {X_train_smote.shape[0]:,} muestras")
    print(f"      scale_pos_weight = {scale_pos_weight:.2f} (neg/pos = {n_neg}/{n_pos})")

    # =================================================================
    # 6. ENTRENAMIENTO XGBoost
    # =================================================================
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    print(f"\n[6/6] Entrenando XGBoost...")
    print(f"      n_estimators={xgb_model.n_estimators}, "
          f"max_depth={xgb_model.max_depth}, "
          f"lr={xgb_model.learning_rate}")

    xgb_model.fit(
        X_train_smote,
        y_train_smote,
        eval_set=[(X_val_prep, y_val)],
        verbose=False,
    )

    # =================================================================
    # 7. EVALUACION SOBRE TEST SET
    # =================================================================
    y_pred = xgb_model.predict(X_test_prep)
    y_prob = xgb_model.predict_proba(X_test_prep)[:, 1]

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n[7/7] Evaluacion sobre Test Set ({len(X_test):,} muestras):")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=["Cumple SLA", "Incumple SLA"]))
    print("-" * 60)
    print(f"      Recall (Incumple) : {recall:.4f}")
    print(f"      Precision         : {precision:.4f}")
    print(f"      F1-Score          : {f1:.4f}")
    print(f"      ROC-AUC           : {auc:.4f}")

    # =================================================================
    # EXPORTACION DE ARTEFACTOS -> backend/
    # =================================================================
    os.makedirs(BACKEND_DIR, exist_ok=True)

    joblib.dump(xgb_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROC_PATH)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  ARTEFACTOS EXPORTADOS A backend/")
    print(f"{'=' * 60}")
    print(f"  -> {MODEL_PATH}")
    print(f"  -> {PREPROC_PATH}")
    print(f"\n  Tiempo total: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
