# Prediccion de Incumplimiento de SLA en ITSM

Este proyecto implementa una solucion completa para predecir el riesgo de incumplimiento de SLA en tickets de soporte tecnico utilizando Machine Learning y explicabilidad (XAI). 

## Arquitectura

La arquitectura se compone de los siguientes modulos:
- **Modelos de Machine Learning**: Se evaluan **4 modelos** — Regresion Logistica, Random Forest, XGBoost y MLP Deep (128-64-32) — usando SMOTE + `scale_pos_weight` para maximizar el Recall sobre la clase minoritaria. **XGBoost** es el modelo seleccionado para produccion por su mejor balance AUC/Latencia. La variable `tiempo_resolucion_hrs` esta explicitamente excluida en todos los pipelines para evitar *target leakage*.
- **Backend (FastAPI)**: Una API REST de alto rendimiento que expone el modelo predictivo, procesa los inputs de los usuarios mediante un pipeline estructurado y genera valores SHAP para explicabilidad.
- **Explicabilidad (SHAP)**: Analisis en tiempo real de como las caracteristicas del ticket influyen en la prediccion.
- **Frontend (Angular)**: Interfaz de usuario interactiva para ingresar parametros del ticket y visualizar tanto el riesgo predicho (bandas de riesgo) como la justificacion del modelo.

## Estructura del Directorio

```text
/
├── backend/            # API en FastAPI + artefactos .pkl del modelo
│   ├── main.py         # Servidor FastAPI (uvicorn)
│   ├── modelo_xgboost.pkl
│   ├── preprocessor.pkl
│   └── requirements.txt
├── datasets/           # Datasets CSV generados y procesados
├── docs/               # Documentacion y graficas para informes
│   └── imagenes_informe/
├── frontend/           # Proyecto Angular 17+
├── notebooks/          # Jupyter Notebooks con EDA, entrenamiento y evaluacion
├── scripts/            # Scripts auxiliares de visualizacion
├── run_ml.py           # Pipeline de entrenamiento de extremo a extremo
├── .gitignore
└── README.md
```

## Reproduccion Completa del Flujo (4 comandos)

Los siguientes 4 comandos reproducen el flujo completo desde cero: entrenar el modelo, instalar dependencias del backend, levantar la API y arrancar el frontend.

> **Prerequisitos**: Python 3.10+, Node.js 18+, npm. Ejecutar desde la raiz del proyecto.

### Comando 1 — Entrenar el modelo y exportar artefactos

```bash
py run_ml.py
```

Carga `datasets/dataset_tickets_con_senal.csv`, elimina `tiempo_resolucion_hrs` (anti-leakage), entrena XGBoost con SMOTE + pesos de clase, imprime el classification report sobre Test, y exporta `modelo_xgboost.pkl` + `preprocessor.pkl` directamente a `backend/`.

> **Nota**: Para una comparativa completa de los **4 modelos** (Regresion Logistica, Random Forest, XGBoost, MLP Deep 128-64-32), ejecuta el notebook `notebooks/modelos_ticketera.ipynb` con *Restart & Run All*.

### Comando 2 — Instalar dependencias del backend

```bash
pip install -r backend/requirements.txt
```

Instala las versiones exactas de todas las librerias necesarias (FastAPI, XGBoost, scikit-learn, SHAP, etc.).

### Comando 3 — Levantar el backend (API REST)

```bash
uvicorn backend.main:app --reload
```

Inicia el servidor FastAPI en `http://localhost:8000`. La API carga automaticamente los artefactos `.pkl` exportados en el paso 1 y expone el endpoint `/predict` para predicciones con explicabilidad SHAP.

### Comando 4 — Levantar el frontend (Angular)

```bash
cd frontend && npm install && ng serve
```

Instala dependencias de Node e inicia la aplicacion Angular en `http://localhost:4200`. La interfaz se conecta al backend para enviar tickets y visualizar predicciones de riesgo.

## Stack Tecnologico

| Capa | Tecnologia | Version |
|------|-----------|---------|
| Modelos ML | Regresion Logistica, Random Forest, **XGBoost** (prod) | sklearn 1.9.0 / XGBoost 3.4.1 |
| Modelo DL | MLP Deep 128-64-32 (TensorFlow/Keras) | TensorFlow 2.x |
| ML Pipeline | scikit-learn | 1.9.0 |
| Balanceo | imbalanced-learn (SMOTE) | 0.14.2 |
| Explicabilidad | SHAP | 0.52.0 |
| Backend | FastAPI + Uvicorn | 0.110.1 |
| Frontend | Angular | 17+ |
| Datos | Pandas + NumPy | 2.3.2 |
