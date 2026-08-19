# Predicción de Incumplimiento de SLA en ITSM

Este proyecto implementa una solución completa para predecir el riesgo de incumplimiento de SLA en tickets de soporte técnico utilizando Machine Learning y explicabilidad (XAI). 

## Arquitectura

La arquitectura se compone de los siguientes módulos:
- **Modelo de Machine Learning**: XGBClassifier (XGBoost) entrenado con técnicas de balanceo (SMOTE) para lidiar con el desbalanceo de clases natural en el incumplimiento de SLAs.
- **Backend (FastAPI)**: Una API REST de alto rendimiento que expone el modelo predictivo, procesa los inputs de los usuarios mediante un pipeline estructurado y genera valores SHAP para explicabilidad.
- **Explicabilidad (SHAP)**: Análisis en tiempo real de cómo las características del ticket influyen en la predicción.
- **Frontend (Angular)**: Interfaz de usuario interactiva para ingresar parámetros del ticket y visualizar tanto el riesgo predicho (bandas de riesgo) como la justificación del modelo.

## Estructura del Directorio

```text
/
├── backend/            # API en FastAPI, modelos exportados y scripts de análisis
├── datasets/           # Datasets CSV generados y procesados
├── docs/               # Documentación y gráficas generadas para informes (imágenes)
├── frontend/           # Proyecto Angular 17+
├── notebooks/          # Jupyter Notebooks con EDA, entrenamiento y evaluación
├── .gitignore          # Archivos excluidos del control de versiones
└── README.md           # Este archivo
```

## Instrucciones de Instalación y Ejecución

### 1. Levantar el Backend (FastAPI)

Navega al directorio `backend` e instala las dependencias. Se recomienda usar un entorno virtual.

```bash
cd backend
python -m venv venv
# Activar entorno (Windows)
venv\Scripts\activate
# Activar entorno (Mac/Linux)
# source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```
La API estará disponible en `http://localhost:8000`.

### 2. Levantar el Frontend (Angular)

Navega al directorio `frontend` e instala las dependencias de Node:

```bash
cd frontend
npm install
ng serve
```
La aplicación web estará disponible en `http://localhost:4200`.
