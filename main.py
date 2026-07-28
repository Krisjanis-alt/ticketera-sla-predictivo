from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Instrucciones para levantar el servidor localmente:
# uvicorn main:app --reload

app = FastAPI()

# Configuración de CORS para permitir peticiones desde cualquier origen (Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga del modelo al inicio de la aplicación
try:
    model = joblib.load("modelo_xgboost.pkl")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Esquema de datos de entrada
class TicketInput(BaseModel):
    Categoria: str
    Prioridad: str
    Seniority: str
    Hora_Creacion: int

@app.post("/predict")
def predict(ticket: TicketInput):
    if model is None:
        return {"error": "El modelo no se encuentra cargado correctamente."}

    # Transformar los datos de entrada a un DataFrame
    df_input = pd.DataFrame([ticket.model_dump()])

    # Aplicar One-Hot Encoding al input
    df_encoded = pd.get_dummies(df_input)

    # Obtener las columnas que el modelo vio durante el entrenamiento
    # xgboost suele guardar los nombres de las features en 'feature_names_in_' o similar
    try:
        expected_cols = model.feature_names_in_
    except AttributeError:
        # Si el modelo no guardó los nombres, esta parte requeriría la lista estática de columnas
        expected_cols = df_encoded.columns # Fallback, puede fallar si faltan columnas

    # Rellenar con ceros las columnas faltantes y mantener el orden correcto
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Asegurar el mismo orden de columnas que el modelo espera
    df_encoded = df_encoded[expected_cols]

    # Obtener la probabilidad de incumplimiento (asumiendo que la clase 1 es "incumplimiento")
    # predict_proba retorna un array con probabilidades para cada clase [[prob_0, prob_1]]
    probabilidad = model.predict_proba(df_encoded)[0][1]
    prob_porcentaje = probabilidad * 100

    # Clasificar el riesgo según la probabilidad
    if prob_porcentaje <= 30:
        nivel_riesgo = "Bajo"
    elif prob_porcentaje <= 70:
        nivel_riesgo = "Medio"
    else:
        nivel_riesgo = "Alto"

    return {
        "riesgo": nivel_riesgo,
        "probabilidad": round(prob_porcentaje, 2)
    }
