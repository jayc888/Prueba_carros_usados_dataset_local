import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost as xgb 
import os # Necesario para manejar rutas de archivos

# ====================================================================
#              PARTE 1: CARGA DE DATOS Y MODELO (RUTAS CORREGIDAS)
# ====================================================================

# Ruta del modelo: Desde src/app.py, sube un nivel (..) y entra a models/
MODEL_PATH = "../models/hiper_xgboost_50iter_42.json" 

try:
    # Carga del modelo XGBoost nativo
    final_model = xgb.Booster()
    final_model.load_model(MODEL_PATH)
    
    # --- Carga de variables de codificación y estructura ---
    # Los archivos .joblib están en la misma carpeta que app.py (src/), así que usamos la ruta directa.

    X_train_columns = joblib.load('X_train_columns.joblib')
    manufacturer_target_encoding = joblib.load('manufacturer_target_encoding.joblib')
    paint_color_target_encoding = joblib.load('paint_color_target_encoding.joblib')
    state_target_encoding = joblib.load('state_target_encoding.joblib')
    
    # Cargar las listas de valores únicos
    CylindersUnique = joblib.load('CylindersUnique.joblib')
    FuelUnique = joblib.load('FuelUnique.joblib')
    TransmissionUnique = joblib.load('TransmissionUnique.joblib')
    DriveUnique = joblib.load('DriveUnique.joblib')
    SizeUnique = joblib.load('SizeUnique.joblib')
    
    # Cargar el año actual (si lo guardaste)
    try:
        ANO_ACTUAL = joblib.load('ANO_ACTUAL.joblib')
    except:
        # Valor de fallback si el archivo ANO_ACTUAL.joblib no existe o falla
        ANO_ACTUAL = 2024 

    # Obtener las listas de las claves de los diccionarios de encoding
    ManufacturerUnique = list(manufacturer_target_encoding.keys())
    PaintColorUnique = list(paint_color_target_encoding.keys())
    StateUnique = list(state_target_encoding.keys())
    
except FileNotFoundError as e:
    # Mensaje de error ajustado para reflejar que busca los archivos en el directorio actual (src/)
    st.error(f"""
        Error: No se pudo cargar un archivo necesario. 
        Asegúrate de que el modelo y TODOS los archivos .joblib se encuentren 
        en las ubicaciones correctas.

        - Modelo esperado: {os.path.abspath(MODEL_PATH)}
        - Archivos .joblib esperados: {os.path.join(os.getcwd(), str(e).split(': ')[-1])}
        
        Falta: {e}
    """)
    st.stop()
except xgb.core.XGBoostError as e:
    st.error(f"Error al cargar el modelo XGBoost desde {MODEL_PATH}: {e}")
    st.stop()


# ====================================================================
#              PARTE 2: FUNCIÓN DE PREDICCIÓN Y ENCODING
# ====================================================================

@st.cache_data
def codificar_datos_para_modelo(odometer, year, manufacturer, paint_color, state, cylinders, fuel, transmission, drive, size, X_train_columns):
    """
    Procesa los inputs del usuario para crear un DMatrix listo para la predicción del modelo XGBoost,
    replicando la lógica exacta de la función solicitar_datos_usuario().
    """
    age = ANO_ACTUAL - year

    # 1. ENCODING (Target/Media Encoding)
    manufacturer_encoded = manufacturer_target_encoding.get(manufacturer, 0)
    paint_color_encoded = paint_color_target_encoding.get(paint_color, 0)
    state_encoded = state_target_encoding.get(state, 0)

    # Inicializar el diccionario base con 0 para todas las columnas de OHE que existen
    datos = {col: 0 for col in X_train_columns}
    
    # Rellenar datos numéricos y target encoded
    datos.update({
        'odometer': odometer,
        'age': age,
        'manufacturer_encoded': manufacturer_encoded,
        'paint_color_encoded': paint_color_encoded,
        'state_encoded': state_encoded,
    })

    # Función de ayuda para activar la columna OHE (True/False se codifican a 1/0 en DataFrame)
    def activar_ohe(feature_name, selected_value):
        col_name = f'{feature_name}_{selected_value}'
        if col_name in datos:
             datos[col_name] = 1 
    
    # 2. ACTIVACIÓN DE ONE-HOT ENCODING (OHE)
    
    # Cylinders: Incluye 'other'
    activar_ohe('cylinders', cylinders)
    
    # Fuel: Incluye 'other'
    activar_ohe('fuel', fuel)
    
    # Transmission: 'automatic' no tiene columna (es la base)
    if transmission != 'automatic':
        activar_ohe('transmission', transmission)
    
    # Drive: 'awd' no tiene columna (es la base)
    if drive != 'awd':
        activar_ohe('drive', drive)
    
    # Size: Todas tienen columna
    activar_ohe('size', size)
    
    # Convertir a DataFrame y asegurar el orden de las columnas
    X_nuevo = pd.DataFrame([datos])[X_train_columns]
    
    # El modelo Booster nativo de XGBoost requiere un DMatrix
    X_dmatrix = xgb.DMatrix(X_nuevo)
    
    return X_dmatrix


# ====================================================================
#                      PARTE 3: INTERFAZ STREAMLIT
# ====================================================================

st.set_page_config(
    page_title="Auto Geeks: Predicción de Precio",
    layout="wide"
)

# --- Título y Encabezado ---
st.title("🚗 Auto Geeks: ayudándote a ponerle precio a tu vehículo 🤑")
st.markdown("---")
st.markdown("Utiliza los siguientes campos para ingresar las características de tu vehículo y obtener un precio estimado.")

col1, col2 = st.columns(2)

with col1:
    st.header("🔢 Datos Numéricos")
    
    # Odometer (Slider)
    odometer = st.slider(
        "**Odometer** (millas):",
        min_value=0,
        max_value=300000,
        value=50000,
        step=500
    )
    
    # Año (Slider)
    year = st.slider(
        "**Año** del vehículo:",
        min_value=1980,
        max_value=ANO_ACTUAL,
        value=ANO_ACTUAL - 5,
        step=1
    )
    
    # Manufacturer (Selectbox)
    manufacturer = st.selectbox(
        "**Manufacturer**:",
        options=ManufacturerUnique,
        index=0,
        format_func=lambda x: x.title()
    ).lower()

    # State (Selectbox)
    state = st.selectbox(
        "**State** (Estado de registro):",
        options=StateUnique,
        index=0,
        format_func=lambda x: x.upper()
    ).lower()

with col2:
    st.header("⚙️ Especificaciones del Vehículo")
    
    # Paint Color (Selectbox)
    paint_color = st.selectbox(
        "**Paint Color**:",
        options=PaintColorUnique,
        index=0,
        format_func=lambda x: x.title()
    ).lower()
    
    # Cylinders (Radio Button)
    default_cyl = '6 cylinders' if '6 cylinders' in CylindersUnique else CylindersUnique[0]
    cylinders = st.radio(
        "**Cylinders**:",
        options=CylindersUnique,
        index=CylindersUnique.index(default_cyl),
        horizontal=True
    ).lower()

    # Fuel (Radio Button)
    default_fuel = 'gas' if 'gas' in FuelUnique else FuelUnique[0]
    fuel = st.radio(
        "**Fuel**:",
        options=FuelUnique,
        index=FuelUnique.index(default_fuel),
        horizontal=True
    ).lower()
    
    # Transmission (Selectbox)
    default_trans = 'automatic' if 'automatic' in TransmissionUnique else TransmissionUnique[0]
    transmission = st.selectbox(
        "**Transmission**:",
        options=TransmissionUnique,
        index=TransmissionUnique.index(default_trans),
        format_func=lambda x: x.title()
    ).lower()

    # Drive (Radio Button)
    default_drive = 'fwd' if 'fwd' in DriveUnique else DriveUnique[0]
    drive = st.radio(
        "**Drive**:",
        options=DriveUnique,
        index=DriveUnique.index(default_drive),
        horizontal=True
    ).lower()

    # Size (Selectbox)
    default_size = 'mid-size' if 'mid-size' in SizeUnique else SizeUnique[0]
    size = st.selectbox(
        "**Size** (Tamaño/Categoría):",
        options=SizeUnique,
        index=SizeUnique.index(default_size),
        format_func=lambda x: x.replace('-', ' ').title()
    ).lower()
    
st.markdown("---")

# --- Botón de Predicción ---
if st.button("Calcular Precio Estimado", type="primary", use_container_width=True):
    with st.spinner('Realizando cálculo de predicción...'):
        try:
            # 1. Codificar los datos
            X_dmatrix = codificar_datos_para_modelo(
                odometer=odometer, year=year, manufacturer=manufacturer, paint_color=paint_color,
                state=state, cylinders=cylinders, fuel=fuel, transmission=transmission,
                drive=drive, size=size, X_train_columns=X_train_columns
            )
            
            # 2. Realizar la predicción
            precio_estimado = final_model.predict(X_dmatrix)[0]

            # 3. Mostrar el resultado
            st.success("✅ Predicción completada")
            st.balloons()
            
            st.markdown(
                f"""
                <div style="background-color: #e0f7fa; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: #00796b;">El valor estimado del auto es:</h2>
                    <h1 style="color: #004d40; font-size: 3em;">${precio_estimado:,.2f}</h1>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")
            st.info("Asegúrese de que los datos ingresados sean válidos y que las variables de codificación se hayan cargado correctamente.")

