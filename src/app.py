import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
# Importamos 'pathlib' para manejar rutas de archivos de forma segura
from pathlib import Path

# --- Configuración de la página ---
st.set_page_config(
    page_title="Predictor de Precio de Vehículos",
    page_icon="🚗",
    layout="wide"
)

# --- Título principal ---
st.title("🚗 Predictor de Precio de Vehículos")
st.markdown("""
Esta aplicación predice el valor de mercado de un vehículo basado en sus características.
Complete la información solicitada y haga clic en **Predecir Precio**.
""")

# --- Sidebar para información adicional ---
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    **Instrucciones:**
    1. Complete todos los campos del formulario
    2. Use los valores sugeridos en cada categoría
    3. Haga clic en 'Predecir Precio'
    4. Vea el resultado en la parte inferior
    """)

# Obtener el directorio base del script actual (src)
# Esto asegura que las rutas de los archivos sean correctas sin importar
# desde dónde se ejecute el comando 'streamlit run'.
BASE_DIR = Path(__file__).resolve().parent

# --- Cargar los artefactos guardados ---
@st.cache_resource
def cargar_artefactos():
    """Carga todos los archivos del modelo y artefactos de pre-procesamiento."""
    try:
        # Definir una función auxiliar para cargar archivos usando la ruta base
        def load_artifact(filename):
            """Carga un artefacto de joblib o el modelo con su ruta correcta."""
            filepath = BASE_DIR / filename
            if filename.endswith('.json'):
                # Cargar el modelo XGBoost (el .json)
                modelo = xgb.Booster()
                modelo.load_model(str(filepath))
                return modelo
            else:
                # Cargar archivos joblib
                return joblib.load(filepath)

        # Cargar listas de categorías únicas
        CylindersUnique = load_artifact('CylindersUnique.joblib')
        FuelUnique = load_artifact('FuelUnique.joblib')
        TransmissionUnique = load_artifact('TransmissionUnique.joblib')
        # NOTA: Si este archivo contiene ['fwd', 'rwd', 'awd'], debe ser regenerado
        # para contener ['fwd', 'rwd', '4wd'] si ese es el valor correcto.
        DriveUnique = load_artifact('DriveUnique.joblib') 
        SizeUnique = load_artifact('SizeUnique.joblib')
        
        # Cargar diccionarios de target encoding
        manufacturer_target_encoding = load_artifact('manufacturer_target_encoding.joblib')
        paint_color_target_encoding = load_artifact('paint_color_target_encoding.joblib')
        state_target_encoding = load_artifact('state_target_encoding.joblib')
        
        # Cargar columnas del modelo
        X_train_columns = load_artifact('X_train_columns.joblib')
        
        # Cargar año actual
        ANO_ACTUAL = load_artifact('ANO_ACTUAL.joblib')
        
        # Cargar modelo (usamos la función auxiliar que ya maneja el .json)
        modelo = load_artifact('hiper_xgboost_50iter_42.json')
        
        st.success("Archivos del modelo cargados exitosamente. 🎉")
        
        return {
            'CylindersUnique': CylindersUnique,
            'FuelUnique': FuelUnique,
            'TransmissionUnique': TransmissionUnique,
            'DriveUnique': DriveUnique,
            'SizeUnique': SizeUnique,
            'manufacturer_target_encoding': manufacturer_target_encoding,
            'paint_color_target_encoding': paint_color_target_encoding,
            'state_target_encoding': state_target_encoding,
            'X_train_columns': X_train_columns,
            'ANO_ACTUAL': ANO_ACTUAL,
            'modelo': modelo
        }
    except FileNotFoundError as e:
        # Se muestra un mensaje de error más específico y se detiene la ejecución
        st.error(f"❌ Error crítico: No se encontró el archivo '{Path(str(e).split(':')[-1].strip()).name}'. Asegúrese de que todos los archivos `.joblib` y el modelo estén en la misma carpeta que `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar los archivos del modelo: {e}")
        return None

# --- Cargar artefactos ---
artefactos = cargar_artefactos()

if artefactos is None:
    # st.stop() ya se llama dentro de cargar_artefactos si hay FileNotFoundError
    # pero lo mantenemos para cualquier otro error de carga
    if not st.session_state.get('load_error', False):
           st.stop()


# --- Extraer artefactos ---
CylindersUnique = artefactos['CylindersUnique']
FuelUnique = artefactos['FuelUnique']
TransmissionUnique = artefactos['TransmissionUnique']
DriveUnique = artefactos['DriveUnique']
SizeUnique = artefactos['SizeUnique']
manufacturer_target_encoding = artefactos['manufacturer_target_encoding']
paint_color_target_encoding = artefactos['paint_color_target_encoding']
state_target_encoding = artefactos['state_target_encoding']
X_train_columns = artefactos['X_train_columns']
ANO_ACTUAL = artefactos['ANO_ACTUAL']
modelo = artefactos['modelo']

# --- Formulario de entrada de datos ---
st.header("📝 Información del Vehículo")

# Dividir en columnas para mejor organización
col1, col2 = st.columns(2)

with col1:
    st.subheader("Características Principales")
    
    # Obtener listas únicas para los selectbox
    manufacturer_options = list(manufacturer_target_encoding.keys())
    paint_color_options = list(paint_color_target_encoding.keys())
    state_options = list(state_target_encoding.keys())
    
    odometer = st.number_input(
        "Odometer (millas)",
        min_value=0,
        max_value=500000,
        value=50000,
        step=1000,
        help="Kilometraje actual del vehículo"
    )
    
    year = st.number_input(
        "Año del vehículo",
        min_value=1980,
        max_value=ANO_ACTUAL,
        value=2020,
        step=1
    )
    
    manufacturer = st.selectbox(
        "Fabricante",
        options=manufacturer_options,
        help="Seleccione el fabricante del vehículo"
    )
    
    paint_color = st.selectbox(
        "Color",
        options=paint_color_options
    )
    
    state = st.selectbox(
        "Estado",
        options=state_options,
        help="Estado donde se encuentra el vehículo"
    )

with col2:
    st.subheader("Especificaciones Técnicas")
    
    cylinders = st.selectbox(
        "Cilindros",
        options=CylindersUnique
    )
    
    fuel = st.selectbox(
        "Combustible",
        options=FuelUnique
    )
    
    transmission = st.selectbox(
        "Transmisión",
        options=TransmissionUnique
    )
    
    drive = st.selectbox(
        "Tracción",
        options=DriveUnique
    )
    
    size = st.selectbox(
        "Tamaño",
        options=SizeUnique
    )

# Calcular edad del vehículo
age = ANO_ACTUAL - year

# --- Mostrar resumen de la selección ---
st.header("📊 Resumen de la Selección")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Kilometraje", f"{odometer:,} millas")
    st.metric("Año", year)
    st.metric("Edad", f"{age} años")

with col2:
    st.metric("Fabricante", manufacturer)
    st.metric("Color", paint_color)
    st.metric("Estado", state)

with col3:
    st.metric("Cilindros", cylinders)
    st.metric("Combustible", fuel)
    st.metric("Transmisión", transmission)
    st.metric("Tracción", drive)
    st.metric("Tamaño", size)

# --- Función para preparar los datos ---
def preparar_datos():
    """Prepara los datos de entrada del usuario para la predicción del modelo."""
    # Target Encoding
    # Se usa .get(key, 0) para manejar valores que no estén en el diccionario (out-of-vocabulary)
    manufacturer_encoded = manufacturer_target_encoding.get(manufacturer, 0)
    paint_color_encoded = paint_color_target_encoding.get(paint_color, 0)
    state_encoded = state_target_encoding.get(state, 0)

    # One-Hot Encoding para variables categóricas
    
    # cylinders
    cylinders_dict = {f'cylinders_{c}': 0 for c in CylindersUnique}
    if cylinders in CylindersUnique:
        cylinders_dict[f'cylinders_{cylinders}'] = 1

    # fuel
    fuel_dict = {f'fuel_{f}': 0 for f in FuelUnique}
    if fuel in FuelUnique:
        fuel_dict[f'fuel_{fuel}'] = 1

    # transmission (excluyendo 'automatic' como base)
    transmission_dict = {f'transmission_{t}': 0 for t in TransmissionUnique if t != 'automatic'}
    if transmission in TransmissionUnique and transmission != 'automatic':
        transmission_dict[f'transmission_{transmission}'] = 1

    # drive (excluyendo '4wd' como base si ese fue el valor original)
    # NOTA: Asumimos que la categoría omitida es '4wd' para tu modelo.
    # Si '4wd' está en DriveUnique, esta línea funcionará correctamente.
    drive_dict = {f'drive_{d}': 0 for d in DriveUnique if d != '4wd'} # <-- CORRECCIÓN: CAMBIADO DE 'awd' a '4wd'
    if drive in DriveUnique and drive != '4wd': # <-- CORRECCIÓN: CAMBIADO DE 'awd' a '4wd'
        drive_dict[f'drive_{drive}'] = 1

    # size
    size_dict = {f'size_{s}': 0 for s in SizeUnique}
    if size in SizeUnique:
        size_dict[f'size_{s}'] = 1

    # Construir diccionario final con todas las características
    datos = {
        'odometer': odometer,
        'age': age,
        'manufacturer_encoded': manufacturer_encoded,
        'paint_color_encoded': paint_color_encoded,
        'state_encoded': state_encoded,
    }
    datos.update(cylinders_dict)
    datos.update(fuel_dict)
    datos.update(transmission_dict)
    datos.update(drive_dict)
    datos.update(size_dict)
    
    return datos

# --- Botón para realizar predicción ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predecir_button = st.button(
        "🎯 Predecir Precio",
        use_container_width=True,
        type="primary"
    )

if predecir_button:
    try:
        with st.spinner("Calculando precio estimado..."):
            # Preparar datos
            datos_usuario = preparar_datos()
            
            # Convertir a DataFrame
            X_nuevo = pd.DataFrame([datos_usuario])
            
            # Asegurar mismo orden de columnas que el modelo fue entrenado
            # Rellenar con 0 si faltan columnas (aunque no debería si los diccionarios son completos)
            # y reordenar según X_train_columns
            X_nuevo = X_nuevo.reindex(columns=X_train_columns, fill_value=0)
            
            # Realizar predicción
            # El modelo XGBoost necesita un DMatrix
            precio_estimado = modelo.predict(xgb.DMatrix(X_nuevo))[0]
        
        # Mostrar resultado
        st.markdown("---")
        st.header("💰 Resultado de la Predicción")
        
        # Crear métrica destacada
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.metric(
                label="**PRECIO ESTIMADO DEL VEHÍCULO**",
                value=f"${precio_estimado:,.2f}",
                delta=None
            )
            
            # Información adicional
            st.success("✅ Predicción completada exitosamente")
            st.info("💡 Este es un valor estimado basado en el modelo de machine learning")
            
    except Exception as e:
        st.error(f"❌ Error al realizar la predicción: {e}")
        st.info("🔍 Verifique que todos los campos estén completos correctamente")

# --- Información adicional en el footer ---
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        color: gray;
        font-size: 0.8em;
    }
    </style>
    <div class="footer">
    Modelo de predicción basado en XGBoost | Desarrollado con Streamlit
    </div>
    """,
    unsafe_allow_html=True
)


