import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

# Cambiado el título de la página
st.set_page_config(page_title="AutoGeeks - Price Prediction", page_icon="🚗", layout="wide")

# ----------------------------
# Diccionarios de Target Encoding (copiados directamente de tu código)
# ----------------------------
manufacturer_target_encoding = {
    'ford': 16714.94783628016, 'honda': 10063.92786534865, 'dodge': 11877.43635250918, 
    'chrysler': 7943.282301845819, 'toyota': 13253.558976332499, 'jeep': 15872.569204152249, 
    'lexus': 15936.267190569744, 'chevrolet': 15145.636053529257, 'bmw': 16010.542547425473, 
    'gmc': 18813.763487475917, 'mercedes-benz': 17310.777483443708, 'mazda': 8995.312252964426, 
    'rover': 21706.88581314879, 'ram': 23947.62212432697, 'nissan': 10512.536777283613, 
    'ferrari': 127963.75, 'audi': 16824.23698630137, 'mitsubishi': 14006.723300970874, 
    'infiniti': 14337.616228070176, 'volkswagen': 9796.543062200957, 'kia': 9808.35274356103, 
    'pontiac': 6461.3747072599535, 'hyundai': 9412.074961360124, 'fiat': 9222.725274725275, 
    'acura': 11247.532859680285, 'cadillac': 12744.04648241206, 'lincoln': 11102.017777777777, 
    'jaguar': 12636.115606936417, 'saturn': 4629.353535353535, 'volvo': 9505.84064665127, 
    'alfa-romeo': 27085.842105263157, 'buick': 9780.650499286734, 'subaru': 11479.211376404495, 
    'mini': 10080.272980501393, 'mercury': 4335.3, 'porsche': 31963.279569892475, 
    'harley-davidson': 15607.08695652174, 'tesla': 46611.782608695656, 'datsun': 12519.0, 
    'land rover': 19276.428571428572, 'aston-martin': 48628.333333333336
}

paint_color_target_encoding = {
    'black': 15585.572984527687, 'blue': 11941.67837404172, 'silver': 11910.346444032159, 
    'white': 17198.32375335921, 'grey': 13918.155058165219, 'yellow': 13800.805128205127, 
    'red': 13166.802085308056, 'green': 9587.626666666667, 'brown': 10861.36651053864, 
    'purple': 8685.955414012738, 'custom': 15374.730678046468, 'orange': 15485.060240963856
}

state_target_encoding = {
    'al': 17635.933734939757, 'ak': 23645.15902140673, 'az': 14714.428235294117, 
    'ar': 15082.485564304461, 'ca': 13963.235571482459, 'co': 16103.822959183673, 
    'ct': 10464.288557213931, 'dc': 10793.262910798123, 'de': 16144.219512195123, 
    'fl': 14618.561689077613, 'ga': 16533.61610878661, 'hi': 18001.47619047619, 
    'id': 17146.21690140845, 'il': 12397.95991091314, 'in': 12069.355555555556, 
    'ia': 13840.637404580153, 'ks': 12560.880085653105, 'ky': 15297.34142394822, 
    'la': 13826.383458646616, 'me': 15576.286036036036, 'md': 14396.27564102564, 
    'ma': 12338.457820738136, 'mi': 14755.86436307375, 'mn': 13575.200870195795, 
    'ms': 11972.735483870967, 'mo': 15408.0964360587, 'mt': 17022.41156462585, 
    'nc': 13006.890919474587, 'ne': 13915.00892857143, 'nv': 18412.357746478872, 
    'nj': 10629.537471612415, 'nm': 15623.05308219178, 'ny': 13413.973400673402, 
    'nh': 12635.791154791155, 'nd': 17868.325396825396, 'oh': 11451.472833408172, 
    'ok': 15557.001265822784, 'or': 13383.070250896057, 'pa': 10888.077563768871, 
    'ri': 10519.433486238531, 'sc': 13923.232876712329, 'sd': 17473.79090909091, 
    'tn': 17772.355146124522, 'tx': 16930.09110688061, 'ut': 19755.518072289156, 
    'vt': 16492.950342465752, 'va': 12375.513888888889, 'wa': 13670.707048458149, 
    'wv': 17278.768, 'wi': 14137.918215613383, 'wy': 20033.904761904763
}

# ----------------------------
# Categorías únicas (basadas en tus diccionarios)
# ----------------------------
ManufacturerUnique = list(manufacturer_target_encoding.keys())
PaintColorUnique = list(paint_color_target_encoding.keys())
StateUnique = list(state_target_encoding.keys())

# Otras categorías (debes completar con tus valores reales)
CylindersUnique = ['6 cylinders', '8 cylinders', '4 cylinders', '5 cylinders', '10 cylinders', '3 cylinders', 'other', '12 cylinders']
FuelUnique = ['gas', 'diesel', 'hybrid', 'electric', 'other']
TransmissionUnique = ['automatic', 'manual', 'other']
DriveUnique = ['rwd', '4wd', 'fwd']
SizeUnique = ['full-size', 'mid-size', 'compact', 'sub-compact']

# Año actual para calcular la edad
ANO_ACTUAL = 2024

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    """Cargar el modelo entrenado"""
    model_paths = [
        "hiper_xgboost_42.pkl", 
        "./models/hiper_xgboost_42.pkl",
        "xgboost_model.pkl",
        "./models/xgboost_model.pkl",
        "final_model.pkl"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                st.success(f"✅ Modelo cargado desde: {model_path}")
                return model
            except Exception as e:
                st.warning(f"No se pudo cargar el modelo en '{model_path}': {e}")
    
    return None

# Cargar modelo
model = load_model()

# ----------------------------
# Images (optional)
# ----------------------------
def load_image(path_options):
    for p in path_options:
        if os.path.exists(p):
            try:
                return Image.open(p)
            except Exception:
                pass
    return None

sidebar_img = load_image(["Pic 1.png", "./assets/Pic 1.png"])
banner_img = load_image(["Pic 2.png", "./assets/Pic 2.png"])

if banner_img:
    st.image(banner_img, use_column_width=True)

# Título Principal Cambiado
st.markdown("<h1 style='text-align: center;'>AutoGeeks</h1>", unsafe_allow_html=True)

# Sidebar
if sidebar_img:
    st.sidebar.image(sidebar_img, use_column_width=True)
st.sidebar.header("Vehicle Features")

# ----------------------------
# User input function (adaptada de tu función solicitar_datos_usuario)
# ----------------------------
def get_user_input():
    # Numerical inputs
    year = st.sidebar.slider('Year', min_value=1900, max_value=2024, value=2018, step=1)
    odometer = st.sidebar.slider('Odometer (millas)', min_value=1, max_value=500000, value=50000, step=1000)
    age = ANO_ACTUAL - year
    
    # Categorical inputs con Target Encoding
    manufacturer = st.sidebar.selectbox('Manufacturer', ManufacturerUnique)
    paint_color = st.sidebar.selectbox('Paint Color', PaintColorUnique)
    state = st.sidebar.selectbox('State', StateUnique)
    cylinders = st.sidebar.selectbox('Cylinders', CylindersUnique)
    fuel = st.sidebar.selectbox('Fuel Type', FuelUnique)
    transmission = st.sidebar.selectbox('Transmission', TransmissionUnique)
    
    # IMPORTANTE: Aquí se usa 'DriveUnique' que sólo incluye 'rwd', '4wd', 'fwd'.
    # Si quieres usar 'awd' y tu modelo lo espera, debes añadir 'awd' a la lista DriveUnique
    # y asegurarte de que tu modelo pueda manejarlo (e.g., que esté en sus feature_names_in_)
    drive = st.sidebar.selectbox('Drive', DriveUnique) 
    
    size = st.sidebar.selectbox('Size', SizeUnique)

    # Aplicar Target Encoding
    manufacturer_encoded = manufacturer_target_encoding.get(manufacturer, 0)
    paint_color_encoded = paint_color_target_encoding.get(paint_color, 0)
    state_encoded = state_target_encoding.get(state, 0)

    # One-Hot Encoding para las demás variables
    cylinders_dict = {f'cylinders_{c}': 1 if c == cylinders else 0 for c in CylindersUnique}
    fuel_dict = {f'fuel_{f}': 1 if f == fuel else 0 for f in FuelUnique}
    transmission_dict = {f'transmission_{t}': 1 if t == transmission else 0 for t in TransmissionUnique}
    drive_dict = {f'drive_{d}': 1 if d == drive else 0 for d in DriveUnique}
    size_dict = {f'size_{s}': 1 if s == size else 0 for s in SizeUnique}

    # Construir el diccionario final
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

    return datos, {
        'year': year,
        'manufacturer': manufacturer,
        'paint_color': paint_color,
        'state': state,
        'cylinders': cylinders,
        'fuel': fuel,
        'transmission': transmission,
        'drive': drive,
        'size': size
    }

# Obtener datos del usuario
user_data, selected_features = get_user_input()

# ----------------------------
# Prepare features for model prediction
# ----------------------------
def prepare_for_prediction(user_data, model):
    """Preparar datos para predicción"""
    # Convertir a DataFrame
    X_nuevo = pd.DataFrame([user_data])
    
    # Si tenemos información de las columnas esperadas por el modelo, reordenar
    if hasattr(model, 'feature_names_in_'):
        expected_columns = model.feature_names_in_
        # Asegurarse de que todas las columnas esperadas estén presentes
        for col in expected_columns:
            if col not in X_nuevo.columns:
                X_nuevo[col] = 0
        X_nuevo = X_nuevo[expected_columns]
    
    return X_nuevo

# ----------------------------
# Layout: two columns
# ----------------------------
left_col, right_col = st.columns(2)

with left_col:
    # Se eliminó la subsección "Valores de Target Encoding"
    st.header("Selected Features") 
    
    # Mostrar características seleccionadas
    st.subheader("🚗 Características Seleccionadas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Year:** {selected_features['year']}")
        st.write(f"**Age:** {user_data['age']} años")
        st.write(f"**Odometer:** {user_data['odometer']:,} millas")
        st.write(f"**Manufacturer:** {selected_features['manufacturer']}")
        st.write(f"**Paint Color:** {selected_features['paint_color']}")
    
    with col2:
        st.write(f"**State:** {selected_features['state']}")
        st.write(f"**Cylinders:** {selected_features['cylinders']}")
        st.write(f"**Fuel Type:** {selected_features['fuel']}")
        st.write(f"**Transmission:** {selected_features['transmission']}")
        st.write(f"**Drive:** {selected_features['drive']}")
        st.write(f"**Size:** {selected_features['size']}")
    
    # El resto del código anterior para Target Encoding fue eliminado.

with right_col:
    st.header("Price Prediction")
    disabled = model is None
    
    if st.button("Predict Vehicle Price", type="primary", disabled=disabled, use_container_width=True):
        try:
            # Preparar datos para predicción
            X_pred = prepare_for_prediction(user_data, model)
            
            # Realizar predicción
            precio_estimado = model.predict(X_pred)[0]
            
            # Mostrar resultado
            st.subheader("Predicted Price")
            st.success(f"Estimated Price: ${precio_estimado:,.2f}")
            
            # Información adicional
            st.subheader("📊 Model Information")
            if hasattr(model, 'feature_names_in_'):
                st.write(f"**Características usadas:** {len(model.feature_names_in_)}")
            st.write(f"**Tipo de modelo:** {type(model).__name__}")
            
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            st.write("### 🐛 Información de Debug")
            st.write(f"**Datos enviados:** {len(user_data)} características")
            if hasattr(model, 'feature_names_in_'):
                st.write(f"**Características esperadas:** {len(model.feature_names_in_)}")
    
    if disabled:
        st.warning("⚠️ No se encontró el modelo entrenado. Sube 'hiper_xgboost_42.pkl' o 'final_model.pkl' para habilitar las predicciones.")

st.markdown("---")
st.caption("AutoGeeks - Using Target Encoding for vehicle price prediction")
