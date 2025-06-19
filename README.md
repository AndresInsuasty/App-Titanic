# 🚢 App-Titanic

Esta aplicación predice la probabilidad de supervivencia de un pasajero del Titanic usando modelos de Machine Learning entrenados con el dataset clásico de Titanic. Incluye una interfaz interactiva construida con Streamlit para que cualquier usuario pueda ingresar sus datos y ver los resultados de tres modelos distintos.

## Características

- **Entrenamiento de modelos de Machine Learning:** Regresión Logística, Bosque Aleatorio y SVC.
- **Preprocesamiento automático de datos:** escalado y codificación de variables categóricas.
- **Interfaz web interactiva:** construida con Streamlit.
- **Resultados visuales:** fáciles de interpretar para el usuario final.

## Estructura del proyecto

```
├── entrenamiento.py         # Script para entrenar y guardar los modelos 
├── my_app.py                # Aplicación Streamlit para predicción interactiva
├── modelos_entrenados/      # Modelos y preprocesador guardados 
│   ├── logistic_regression.joblib 
│   ├── random_forest.joblib 
│   ├── svc.joblib 
│   └── preprocessor.joblib 
├── requirements.txt         # Dependencias del proyecto 
├── README.md                # Este archivo 
└── .gitignore
```


## Instalación

1. Clona el repositorio:
```sh
git clone https://github.com/AndresInsuasty/App-Titanic
cd App-Titanic
```

2. Instala las dependencias:
```sh
pip install -r requirements.txt
```
## Entrenamiento de modelos

Si deseas reentrenar los modelos, ejecuta:
```sh
python entrenamiento.py
```
Esto generará los archivos .joblib en la carpeta `modelos_entrenados/`

## Uso de la aplicación
Lanza la aplicación web con:

```sh
streamlit run my_app.py
```

Abre el enlace que aparece en la terminal para acceder a la interfaz y probar los modelos.

¡Contribuciones y sugerencias son bienvenidas!


Puedes personalizar el enlace del repositorio y agregar detalles adicionales según lo