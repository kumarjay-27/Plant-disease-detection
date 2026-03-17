import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

from utils import preprocess
from disease_info import disease_info

model = tf.keras.models.load_model("model/plant_disease_model.h5")

with open("class_names.json") as f:
    class_names = json.load(f)

st.title("🌿 Plant Disease Detection System")

st.write("Upload a plant leaf image")

file = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

if file:

    img = Image.open(file)

    st.image(img,width=300)

    processed = preprocess(img)

    prediction = model.predict(processed)

    index = np.argmax(prediction)

    confidence = prediction[0][index]

    label = class_names[index]

    st.subheader("Prediction")

    st.write("Disease:",label)

    st.write("Confidence:",round(confidence*100,2),"%")

    st.subheader("Description")

    st.write(disease_info[label]["description"])

    st.subheader("Treatment")

    st.write(disease_info[label]["treatment"])