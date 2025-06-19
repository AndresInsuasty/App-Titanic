import streamlit as st
import joblib
import numpy as np

# Cargar preprocesador y modelos
preprocessor = joblib.load('modelos_entrenados/preprocessor.joblib')
modelo_lr = joblib.load('modelos_entrenados/logistic_regression.joblib')
modelo_rf = joblib.load('modelos_entrenados/random_forest.joblib')
modelo_svc = joblib.load('modelos_entrenados/svc.joblib')

st.title("¿Sobrevivirías al Titanic?")

st.write("Completa tus datos para ver si hubieras sobrevivido según distintos modelos de Machine Learning.")

nombre = st.text_input("¿Cuál es tu nombre?")

age = st.number_input("Edad", min_value=0, max_value=100, value=30)
fare = st.slider("Tarifa pagada (fare)", min_value=0.0, max_value=100.0, value=32.0)
sibsp = st.number_input("Número de hermanos y/o esposos a bordo (sibsp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Número de padres y/o hijos a bordo (parch)", min_value=0, max_value=10, value=0)
sexo_opciones = {'Hombre': 'male', 'Mujer': 'female'}
sexo_seleccionado = st.selectbox("Sexo", options=list(sexo_opciones.keys()))
sex = sexo_opciones[sexo_seleccionado]
pclass = st.selectbox("Clase del boleto (pclass)", options=[1, 2, 3])

if st.button("¿Sobreviviría?"):
    # Crear DataFrame con los datos del usuario
    import pandas as pd
    user_df = pd.DataFrame([{
        'age': age,
        'fare': fare,
        'sibsp': sibsp,
        'parch': parch,
        'sex': sex,
        'pclass': pclass
    }])

    # Preprocesar los datos
    user_processed = preprocessor.transform(user_df)

    # Predecir con cada modelo
    pred_lr = modelo_lr.predict(user_processed)[0]
    pred_rf = modelo_rf.predict(user_processed)[0]
    pred_svc = modelo_svc.predict(user_processed)[0]

    resultados = {
        "Regresión Logística": pred_lr,
        "Bosque Aleatorio": pred_rf,
        "SVC": pred_svc
    }

    st.subheader(f"Resultados para {nombre if nombre else 'el pasajero'}:")

    for modelo, pred in resultados.items():
        if pred == 1:
            st.success(f"{modelo}: ¡Sobrevivirías! 🎉")
        else:
            st.error(f"{modelo}: No sobrevivirías 😢")