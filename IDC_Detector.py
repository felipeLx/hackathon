import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(page_title="Cancer de mama Detector", page_icon="📈")

@st.cache(allow_output_mutation=True)
def load_models():
  model_idc = load_model('models/IDC_model.h5', compile=False)
  model_idc._make_predict_function()
  model_idc.summary()
  session = K.get_session()
  return model_idc, session

st.markdown('<h1 style="text-align: center;">Identificar câncer de mama</h1>', unsafe_allow_html=True)
st.title('Identificação de IDC e Metastases por imagens, usando rede neural, e identificação por variáveis')

c = st.container()
c.title('Identificar IDC')

### load file
uploaded_file = c.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # display image
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        # st.title("Predicted Label for the image is {}".format(map_dict [prediction]))

def IDC_Detector():
    st.sidebar.markdown("# Análise de imagens")

def Metastase_Detector():
    st.markdown("# Identificar Metastases")
    st.sidebar.markdown("# Análise de imagens")

def Variaveis_Detector():
    st.markdown("# Identificar Câncer de mama")
    st.sidebar.markdown("# Análise de variáveis")

page_names_to_funcs = {
    "Identificar IDC": IDC_Detector,
    "Identificar Metastase": Metastase_Detector,
    "Identificar Câncer de mama": Variaveis_Detector,
}

selected_page = st.sidebar.selectbox("Selecione a Página", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

model, session = load_model()
K.set_session(session)

if __name__ == '__main__':
    run()