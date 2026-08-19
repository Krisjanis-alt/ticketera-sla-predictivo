from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
from typing import List

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

# Carga del modelo y preprocesador al inicio de la aplicación
try:
    model = joblib.load("modelo_xgboost.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
except Exception as e:
    print(f"Error al cargar el modelo o preprocesador: {e}")
    model = None
    preprocessor = None

# Esquema de datos de entrada
class TicketInput(BaseModel):
    Categoria: str
    Prioridad: str
    Seniority: str
    Hora_Creacion: int
    Dia_Semana: str
    Tiempo_Resolucion_hrs: float

class ExplicacionShap(BaseModel):
    feature: str
    impacto: float

class PredictionResponse(BaseModel):
    riesgo: str
    probabilidad: float
    explicacion_shap: List[ExplicacionShap]

@app.post("/predict", response_model=PredictionResponse)
def predict(ticket: TicketInput):
    if model is None or preprocessor is None:
        return {"error": "El modelo o el preprocesador no se encuentran cargados correctamente."}

    # Transformar los datos de entrada a un DataFrame
    input_dict = {
        'categoria': [ticket.Categoria],
        'prioridad': [ticket.Prioridad],
        'seniority_agente': [ticket.Seniority],
        'hora_creacion': [ticket.Hora_Creacion],
        'dia_semana': [ticket.Dia_Semana],
        'tiempo_resolucion_hrs': [ticket.Tiempo_Resolucion_hrs]
    }
    df_input = pd.DataFrame(input_dict)

    # Aplicar el preprocesamiento utilizado durante el entrenamiento
    try:
        X_input_prep = preprocessor.transform(df_input)
        expected_cols = preprocessor.get_feature_names_out()
    except Exception as e:
        print(f"Error en el preprocesamiento: {e}")
        return {"error": "Hubo un problema procesando los datos de entrada."}

    # Obtener la probabilidad de incumplimiento (asumiendo que la clase 1 es "incumplimiento")
    # predict_proba retorna un array con probabilidades para cada clase [[prob_0, prob_1]]
    probabilidad = model.predict_proba(X_input_prep)[0][1]
    prob_porcentaje = probabilidad * 100

    # Clasificar el riesgo según la probabilidad
    if prob_porcentaje <= 30:
        nivel_riesgo = "Bajo"
    elif prob_porcentaje <= 70:
        nivel_riesgo = "Medio"
    else:
        nivel_riesgo = "Alto"

    # Calcular valores SHAP para explicabilidad
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input_prep)
        
        # Extraer los valores SHAP para la instancia actual
        if isinstance(shap_values, list):
            vals = shap_values[1][0]  # Si es una lista, tomar la clase positiva (1)
        else:
            if len(shap_values.shape) == 3:
                vals = shap_values[0, :, 1]
            else:
                vals = shap_values[0]
                
        # Mapear a nombres de características
        feature_impacts = [{"feature": str(col), "impacto": float(val)} for col, val in zip(expected_cols, vals)]
        
        # Ordenar por impacto absoluto y tomar el top 5
        feature_impacts.sort(key=lambda x: abs(x["impacto"]), reverse=True)
        top_5_impacts = feature_impacts[:5]
    except Exception as e:
        print(f"Error calculando SHAP: {e}")
        top_5_impacts = []

    return PredictionResponse(
        riesgo=nivel_riesgo,
        probabilidad=round(prob_porcentaje, 2),
        explicacion_shap=top_5_impacts
    )
