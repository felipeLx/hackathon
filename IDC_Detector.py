import json
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from streamlit.logger import get_logger

st.set_page_config(page_title="Invasive Ductal Carcinoma", page_icon="🕵️‍♀️")

LOGGER = get_logger(__name__)

@st.cache(allow_output_mutation=True)
def loadIDCModel():
  model_idc = load_model('models/IDC_model.h5', compile=False)
  model_idc.summary()
  return model_idc

st.markdown('<h1 style="text-align: center;">Identificar câncer de mama</h1>', unsafe_allow_html=True)
st.subheader('Identificação de IDC e Metastases por imagens, usando rede neural, e identificação por variáveis')

c = st.container()
c.markdown("# Identificar IDC 🕵️‍♀️")

### load file
uploaded_file = c.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # transform image to numpy array
    file_bytes = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(50,50), grayscale = False, interpolation = 'nearest', color_mode = 'rgb', keep_aspect_ratio = False)
    input_arr = tf.keras.preprocessing.image.img_to_array(file_bytes)
    input_arr = np.array([input_arr])
    c.image(file_bytes, channels="RGB")

    Genrate_pred = c.button("Generate Prediction")    
    if Genrate_pred:
        model = loadIDCModel()
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        prediction = probability_model.predict(input_arr)
        dict_pred = {0: 'Benigno/Normal', 1: 'Maligno'}
        result = dict_pred[np.argmax(prediction)]
        value = 0
        if result == 'Benigno/Normal':
            value = str(((prediction[0][0])*100).round(2)) + '%'
        else:
            value = str(((prediction[0][1])*100).round(2)) + '%'
        
        # c.write(value)
        c.metric('Predição', result, delta=value, delta_color='normal')
        # c.write(prediction)
        
def IDC_Detector():
    st.sidebar.markdown("# Análise de imagens 🕵️‍♀️")

def Metastase_Detector():
    st.markdown("# Identificar Metástase 🔬")
    st.sidebar.markdown("# Análise de imagens 🔬")

def Variaveis_Detector():
    st.markdown("# Identificar Câncer de mama 🧬")
    st.sidebar.markdown("# Análise de variáveis 🧬")

page_names_to_funcs = {
    "Identificar IDC": IDC_Detector,
    "Identificar Metastase": Metastase_Detector,
    "Identificar Câncer de mama": Variaveis_Detector,
}

selected_page = st.sidebar.selectbox("Selecione a Página", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()