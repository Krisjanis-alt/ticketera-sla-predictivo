import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

def generar_datos_con_senal(n_muestras=5000, random_state=42):
    np.random.seed(random_state)
    ticket_id = np.arange(1, n_muestras + 1)
    hora_creacion = np.random.randint(0, 24, n_muestras)
    dia_semana = np.random.choice(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'], n_muestras)
    prioridad = np.random.choice(['Baja', 'Media', 'Alta'], n_muestras, p=[0.5, 0.3, 0.2])
    seniority = np.random.choice(['Junior', 'Semi-Senior', 'Senior'], n_muestras, p=[0.4, 0.4, 0.2])
    categoria = np.random.choice(['Hardware', 'Software', 'Redes', 'Acceso'], n_muestras)
    
    tiempo_base = np.random.gamma(shape=2.0, scale=4.0, size=n_muestras)
    modificador_seniority = np.where(seniority == 'Junior', 4.0, np.where(seniority == 'Senior', -2.0, 0))
    modificador_prioridad = np.where(prioridad == 'Baja', 6.0, np.where(prioridad == 'Alta', -2.0, 0))
    
    tiempo_resolucion = tiempo_base + modificador_seniority + modificador_prioridad
    tiempo_resolucion = np.clip(tiempo_resolucion, a_min=0.5, a_max=None)
    
    riesgo = -4.0
    riesgo += np.where(prioridad == 'Baja', 1.5, 0)
    riesgo += np.where(prioridad == 'Alta', -1.5, 0)
    riesgo += np.where(seniority == 'Junior', 1.2, 0)
    riesgo += np.where(seniority == 'Senior', -1.0, 0)
    riesgo += np.where(dia_semana == 'Lunes', 0.8, 0)
    riesgo += (hora_creacion / 24.0) * 1.5
    riesgo += (tiempo_resolucion / 8.0)
    
    probabilidad_incumplimiento = 1 / (1 + np.exp(-riesgo))
    incumple_sla = np.random.binomial(1, probabilidad_incumplimiento)
    
    df = pd.DataFrame({
        'ticket_id': ticket_id,
        'dia_semana': dia_semana,
        'hora_creacion': hora_creacion,
        'categoria': categoria,
        'prioridad': prioridad,
        'seniority_agente': seniority,
        'tiempo_resolucion_hrs': np.round(tiempo_resolucion, 2),
        'incumple_sla': incumple_sla
    })
    
    idx_nulos = np.random.choice(df.index, size=int(n_muestras * 0.05), replace=False)
    df.loc[idx_nulos, 'tiempo_resolucion_hrs'] = np.nan
    return df

# Configurar estilo
sns.set_theme(style="whitegrid")
output_dir = "imagenes_informe"
os.makedirs(output_dir, exist_ok=True)

# 1. Generar datos y preparar splits (ya que no se guardó el CSV antes)
df = generar_datos_con_senal(5000)
X = df.drop(['ticket_id', 'incumple_sla'], axis=1)
y = df['incumple_sla']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 2. Cargar modelo y preprocesador
preprocessor = joblib.load('preprocessor.pkl')
xgb_model = joblib.load('modelo_xgboost.pkl')

# Transformar datos de prueba
X_test_prep = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()
X_test_prep_df = pd.DataFrame(X_test_prep, columns=feature_names)

# --- GRÁFICO 1: Balance de Clases ---
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='incumple_sla', data=df, palette='viridis')
plt.title('EDA - Balance de Clases (SLA Incumplido)', fontsize=14)
plt.xlabel('Incumple SLA (0 = No, 1 = Sí)', fontsize=12)
plt.ylabel('Cantidad de Tickets', fontsize=12)

total = len(df)
for p in ax.patches:
    height = p.get_height()
    percentage = f'{100 * height / total:.1f}%'
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'balance_clases.png'), dpi=300)
plt.close()

# --- GRÁFICO 2: Mapa de Calor de Correlación ---
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=[np.number]).drop(['ticket_id'], axis=1, errors='ignore')
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('EDA - Matriz de Correlación (Variables Numéricas)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_correlacion.png'), dpi=300)
plt.close()

# --- GRÁFICO 3: Matriz de Confusión ---
y_pred = xgb_model.predict(X_test_prep)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cumple', 'Incumple'], 
            yticklabels=['Cumple', 'Incumple'])
plt.title('Resultados - Matriz de Confusión (XGBoost)', fontsize=14)
plt.xlabel('Predicción', fontsize=12)
plt.ylabel('Valor Real', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'matriz_confusion.png'), dpi=300)
plt.close()

# --- GRÁFICO 4: SHAP Summary Plot ---
plt.figure(figsize=(10, 8))
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_prep_df)

shap.summary_plot(shap_values, X_test_prep_df, show=False)
plt.title('Explicabilidad - SHAP Summary Plot', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary.png'), bbox_inches='tight', dpi=300)
plt.close()

print(f"Los 4 gráficos han sido generados y guardados en la carpeta '{output_dir}'.")
