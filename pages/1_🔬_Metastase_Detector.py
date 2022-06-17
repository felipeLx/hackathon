import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model

st.set_page_config(page_title="Metastatic Cancer", page_icon="🔬")
st.sidebar.header("# Análise de imagens 🔬")

model_met = load_model('models/Metastatic_model.h5')

# @st.cache(allow_output_mutation=True)
# def loadMetModel():
#  model_met = load_model('pages/models/Metastatic_model.h5', compile=False)
#  model_met.summary()
#  return model_met

c = st.container()
c.markdown('# Identificar Metástase 🔬')

### load file
uploaded_file = c.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # transform image to numpy array
    file_bytes = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(96,96), grayscale = False, interpolation = 'nearest', color_mode = 'rgb', keep_aspect_ratio = False)
    input_arr = tf.keras.preprocessing.image.img_to_array(file_bytes)
    input_arr = np.array([input_arr])
    c.image(file_bytes, channels="RGB")
    
    Genrate_pred = c.button("Gerar Predição")
    if Genrate_pred:
        model = model_met
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        prediction = probability_model.predict(input_arr)
        dict_pred = {0: 'Benigno/Normal', 1: 'Maligno'}
        result = dict_pred[np.argmax(prediction)]
        value = 0
        if result == 'Benigno/Normal':
            value = str(((prediction[0][0])*100).round(2)) + '%'
        else:
            value = str(((prediction[0][1])*100).round(2)) + '%'
        
        c.metric('Predição', result, delta=value, delta_color='normal')