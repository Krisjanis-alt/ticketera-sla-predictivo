import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
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

df_tickets = generar_datos_con_senal(5000)
X = df_tickets.drop(['ticket_id', 'incumple_sla'], axis=1)
y = df_tickets['incumple_sla']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

num_features = ['hora_creacion', 'tiempo_resolucion_hrs']
cat_features = ['dia_semana', 'categoria', 'prioridad', 'seniority_agente']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

X_train_prep = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)

joblib.dump(xgb_model, 'modelo_xgboost.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Modelos generados y exportados!")
