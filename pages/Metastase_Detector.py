import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

@st.cache(allow_output_mutation=True)
def loadMetModel():
  model_met = load_model('../models/Metastase_model.h5', compile=False)
  model_met._make_predict_function()
  model_met.summary()
  session = K.get_session()
  return model_met, session

c = st.container()
c.title('Identificar Metastase')

model = tf.keras.models.load_model("../models/Metastatic_model.h5", compile=False)

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

model_met, session = loadMetModel()
K.set_session(session)