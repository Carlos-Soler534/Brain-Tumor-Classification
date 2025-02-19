import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models
from keras import preprocessing
from PIL import Image
import io

# Cargar los modelos previamente entrenados
@st.cache_resource
def load_models():
    model_cnn = models.load_model("models\CNN_v2.h5")  # Modelo desde cero
    model_tl = models.load_model("models\VGG16.h5")  # Modelo Transfer Learning
    model_ft = models.load_model("models\VGG16_fine_tuning.h5")  # Modelo Fine Tuning
    return {"CNN": model_cnn, "Transfer Learning basado en VGG16": model_tl, "Fine-Tuning basado en VGG16": model_ft}

models = load_models()

# Configuración de la interfaz
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #3a3a3a;
    }
    .result-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("utils/encabezado.jpg", use_container_width=True)
st.title("Diagnóstico de Tumores Cerebrales con IA")
st.sidebar.image("utils/fondo.jpg", use_container_width=True)  # Imagen decorativa en la barra lateral

# Selección del modelo
st.sidebar.header("Seleccionar modelo")
selected_model_name = st.sidebar.selectbox("Elige un modelo para la predicción", list(models.keys()))
selected_model = models[selected_model_name]

# Subir la imagen
st.sidebar.header("Subir imagen de resonancia magnética")
uploaded_file = st.sidebar.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])


# Preprocesamiento de la imagen
def preprocess_image(img):
    img = img.convert("RGB")  
    img = img.resize((224, 224))  # Redimensionar al tamaño de entrada del modelo
    img = preprocessing.image.img_to_array(img)  # Convertir a array
    img = np.expand_dims(img, axis=0)  # Añadir batch
    img = img / 255.0  # Normalizar
    return img

# Realizar predicción si se subió la imagen
if uploaded_file is not None:
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data))
    st.image(img, caption="Imagen Cargada", use_container_width=True)
    processed_img = preprocess_image(img)
    
    # Botón de predicción
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔍 Predecir Diagnóstico"):
            with st.spinner("Analizando la imagen..."):
                prediction = selected_model.predict(processed_img)[0]
                labels = ["Glioma", "Meningioma", "No hay tumor", "Tumor en la glándula pituitaria"]
                result = labels[np.argmax(prediction)]
                probability = np.max(prediction) * 100
                
                # Mostrar resultados
                if result == "No Tumor":
                    st.success(f"✅ **Diagnóstico:** {result}")
                else:
                    st.warning(f"⚠️ **Diagnóstico:** {result}")
                st.info(f"🔎 **Probabilidad:** {probability:.2f}%")
                
                # Mostrar gráfico de predicciones
                fig, ax = plt.subplots()
                ax.bar(["Glioma", "Meningioma", "No hay tumor", "Tumor pituitario"], prediction * 100, color=["blue", "orange", "green", "red"])
                ax.set_ylabel("Probabilidad (%)")
                ax.set_title("Distribución de Predicciones")
                st.pyplot(fig)
