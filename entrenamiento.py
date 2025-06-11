import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# entrenamiento.py


# Crear carpeta para guardar modelos si no existe
os.makedirs('modelos_entrenados', exist_ok=True)

def get_metrics(model, X_tr, y_tr, X_te, y_te):
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_tr, y_pred_train),
            'precision': precision_score(y_tr, y_pred_train),
            'recall': recall_score(y_tr, y_pred_train),
            'f1': f1_score(y_tr, y_pred_train)
        },
        'test': {
            'accuracy': accuracy_score(y_te, y_pred_test),
            'precision': precision_score(y_te, y_pred_test),
            'recall': recall_score(y_te, y_pred_test),
            'f1': f1_score(y_te, y_pred_test)
        }
    }
    return metrics

# Cargar dataset de Titanic desde seaborn
df = sns.load_dataset('titanic')

# Agregar 'pclass' a las variables
variables_cuantitativas = ['age', 'fare', 'sibsp', 'parch']
variables_categoricas = ['sex', 'pclass']  # Agregamos 'pclass' como categórica

df = df[variables_cuantitativas + variables_categoricas + ['survived']]
df = df.dropna()

X = df[variables_cuantitativas + variables_categoricas]
y = df['survived']

# Preprocesamiento: escalado y one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), variables_cuantitativas),
        ('cat', OneHotEncoder(drop='first'), variables_categoricas)
    ]
)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Guardar el preprocesador
joblib.dump(preprocessor, 'modelos_entrenados/preprocessor.joblib')

# Entrenar modelos con los nuevos datos procesados
modelo_lr = LogisticRegression(random_state=42)
modelo_lr.fit(X_train_processed, y_train)
joblib.dump(modelo_lr, 'modelos_entrenados/logistic_regression.joblib')

modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train_processed, y_train)
joblib.dump(modelo_rf, 'modelos_entrenados/random_forest.joblib')

modelo_svc = SVC(probability=True, random_state=42)
modelo_svc.fit(X_train_processed, y_train)
joblib.dump(modelo_svc, 'modelos_entrenados/svc.joblib')

metrics_lr = get_metrics(modelo_lr, X_train_processed, y_train, X_test_processed, y_test)
metrics_rf = get_metrics(modelo_rf, X_train_processed, y_train, X_test_processed, y_test)
metrics_svc = get_metrics(modelo_svc, X_train_processed, y_train, X_test_processed, y_test)

print("\nMétricas de los modelos:")
for name, metrics in zip(
    ["Regresión Logística", "Bosque Aleatorio", "SVC"],
    [metrics_lr, metrics_rf, metrics_svc]
):
    print(f"\n{name}:")
    print(f"  Train - Accuracy: {metrics['train']['accuracy']:.3f}, Precision: {metrics['train']['precision']:.3f}, Recall: {metrics['train']['recall']:.3f}, F1: {metrics['train']['f1']:.3f}")
    print(f"  Test  - Accuracy: {metrics['test']['accuracy']:.3f}, Precision: {metrics['test']['precision']:.3f}, Recall: {metrics['test']['recall']:.3f}, F1: {metrics['test']['f1']:.3f}")