# 📊 Sistema Predictivo de Incumplimiento de SLAs en Tickets de Soporte

Este repositorio contiene el código fuente correspondiente a la Fase 2 del Proyecto para la asignatura. El objetivo principal del sistema es predecir la probabilidad de incumplimiento de los Acuerdos de Nivel de Servicio (SLA) de tickets de soporte técnico, utilizando técnicas de Machine Learning y Deep Learning, e integrando los resultados en una arquitectura web funcional (Backend y Frontend).

---

## 📁 Estructura del Repositorio

El proyecto ha sido diseñado de manera ordenada y modular, dividiéndose en tres componentes principales:

1. Módulo de Machine Learning (modelos_ticketera.ipynb): Contiene el Análisis Exploratorio de Datos (EDA), preprocesamiento (One-Hot Encoding, SMOTE), entrenamiento y validación de los modelos (Random Forest, XGBoost, Multilayer Perceptron) y el análisis de explicabilidad usando SHAP.
2. Módulo de Backend (main.py): Una API REST construida en Python que expone el modelo XGBoost optimizado para recibir nuevos datos y retornar predicciones en tiempo real.
3. Módulo de Frontend (Carpeta Angular): Interfaz web SPA (Single Page Application) donde se pueden ingresar los datos de un ticket nuevo y visualizar el nivel de riesgo mediante un sistema de alertas por colores.

---

## ⚙️ Requisitos Previos

Para ejecutar el proyecto en un entorno local, asegúrate de contar con el siguiente software instalado:

* Python 3.9 o superior.
* Node.js (v14+ recomendado).
* Angular CLI (v14+).

---

## 🚀 Instrucciones de Instalación y Ejecución

Sigue estos pasos en orden para levantar todo el entorno de trabajo:

### Paso 1: Módulo de Machine Learning (Generar el modelo)
1. Abre una terminal en la raíz de este repositorio.
2. Instala las dependencias necesarias de Python ejecutando:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap tensorflow joblib fastapi uvicorn pydantic
3. Abre el archivo modelos_ticketera.ipynb (puedes usar VS Code o Jupyter Notebook).
4. Ejecuta todas las celdas secuencialmente (Run All).
5. Al finalizar, verifica que se haya creado el archivo modelo_xgboost.pkl en tu directorio. Este archivo contiene el modelo entrenado.

### Paso 2: Módulo de Backend (Levantar la API)
1. Asegúrate de que el archivo modelo_xgboost.pkl (creado en el paso anterior) esté en la misma carpeta que el archivo main.py.
2. En la terminal, ejecuta el siguiente comando para levantar el servidor local:
    uvicorn main:app --reload
3. El servidor backend estará corriendo y escuchando peticiones en http://localhost:8000. 
4. (Opcional) Puedes probar los endpoints de la API directamente accediendo a http://localhost:8000/docs (Swagger UI).

### Paso 3: Módulo de Frontend (Levantar la interfaz web)
1. Abre una nueva pestaña o ventana en tu terminal y navega hasta la carpeta del proyecto Angular.
2. Instala las dependencias de Node.js ejecutando:
    npm install
3. Levanta el servidor de desarrollo de Angular con:
    ng serve
4. Abre tu navegador web y dirígete a http://localhost:4200 para visualizar e interactuar con el sistema.

---

## 🛠️ Tecnologías y Librerías Utilizadas

* Data Science & ML: Python, Pandas, NumPy, Scikit-Learn, XGBoost, TensorFlow/Keras, imbalanced-learn (SMOTE), SHAP.
* Backend: FastAPI, Uvicorn, Pydantic, Joblib.
* Frontend: Angular, TypeScript, HTML5, CSS3.

---

## 👤 Autor

* Cristian Aguirre Morales - Desarrollo integral del proyecto (Análisis de Datos, Modelado Predictivo, Implementación Backend y Frontend).
