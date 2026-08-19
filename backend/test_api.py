import requests
import json

url = "http://localhost:8000/predict"

payload = {
    "Categoria": "Software",
    "Prioridad": "Alta",
    "Seniority": "Junior",
    "Hora_Creacion": 9,
    "Dia_Semana": "Lunes",
    "Tiempo_Resolucion_hrs": 12.5
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print("Respuesta exitosa de la API:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error al conectar con la API: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print("Detalle del error:", e.response.text)
