import streamlit as st
import requests
from PIL import Image
import json
import numpy as np
import tensorflow as tf


st.set_page_config(
    page_title="Fire Images Prediction",
    page_icon="🔥🔥🔥",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/rahmad07g',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "Fire Images Prediction By Ragun",
    },
)

tittle = '<h1 style="font-family:sans-serif; color:#000000; text-align:center;">Image Fire Detection</h1>'
st.markdown(tittle, unsafe_allow_html=True)

# Uploader
uploader = st.file_uploader('Upload image from your local...', type=['jpg', 'png'])
if uploader is not None:
    image = Image.open(uploader)
else:
    st.write('waiting for your uploaded image')

URL = "https://fire-detection-ragun.herokuapp.com/v1/models/fire_detection:predict"

# predictor
predictor = st.button('predict')
if predictor: 
    # preprocess input inference image
    image = np.array(image)[:, :, :]
    image = tf.image.resize(image, size=(256, 256))
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    imagelist = image.numpy().tolist()

     # inference image request to backend for model prediction
    
    input_data_json = json.dumps({
                    'signature_name':'serving_default',
                    'instances':imagelist})

    response = requests.post(URL, data=input_data_json)

    if response.status_code == 200:
        res = response.json()
        if res ['predictions'][0][0] > 0.5 :
            text_pred = '<h1 style="font-family:sans-serif; color:#000000; text-align:center;">This is not fire image</h1>'
            st.markdown(text_pred, unsafe_allow_html=True)
        else :
            text_pred = '<h1 style="font-family:sans-serif; color:#ff0000; text-align:center;">This is fire image</h1>'
            st.markdown(text_pred, unsafe_allow_html=True)
    else :
        st.title("Unexpected Error")
    st.image(uploader, use_column_width=True)
        