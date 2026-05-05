import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

import streamlit as st
import joblib
import numpy as np

# Configuración de la página
st.title("Predicción de Distrés Respiratorio")
st.write("Ingrese los parámetros clínicos del paciente:")

# Cargar el modelo entrenado
model = joblib.load('modelo_distres.pkl')

# Crear el formulario de entrada
with st.form("datos_paciente"):
    col1, col2 = st.columns(2)
    
    with col1:
        sexo = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
        driving_pressure = st.number_input("Driving Pressure", value=0.0)
        compliance = st.number_input("Compliance", value=0.0)
        pafio2 = st.number_input("PaFiO2", value=0.0)
        lactato = st.number_input("Lactato", value=0.0)
        rnl = st.number_input("Relación Neutrófilo/Linfocito", value=0.0)

    with col2:
        pcr = st.number_input("PCR", value=0.0)
        pam = st.number_input("PAM", value=0.0)
        urea_crea = st.number_input("Índice Urea/Creatinina", value=0.0)
        fc = st.number_input("Frecuencia Cardíaca", value=0)
        fr = st.number_input("Frecuencia Respiratoria", value=0)

    enviar = st.form_submit_button("Predecir")

if enviar:
    # Organizar los datos para el modelo
    input_data = np.array([[sexo, driving_pressure, compliance, pafio2, lactato, rnl, pcr, pam, urea_crea, fc, fr]])
    prediccion = model.predict(input_data)
    probabilidad = model.predict_proba(input_data)[0][1]

    if prediccion[0] == 1:
        st.error(f"Riesgo de Distrés detectado (Probabilidad: {probabilidad:.2%})")
    else:
        st.success(f"Bajo riesgo de Distrés (Probabilidad: {probabilidad:.2%})")
