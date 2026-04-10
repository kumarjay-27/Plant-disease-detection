import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from utils import preprocess
from disease_info import disease_info

# Page config
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# Load model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

with open("class_names.json") as f:
    class_names = json.load(f)

# Sidebar
st.sidebar.title("🌿 About")
st.sidebar.info(
    "This app detects plant diseases using Deep Learning.\n\n"
    "Upload a leaf image and get instant prediction with treatment."
)

st.sidebar.markdown("---")
st.sidebar.write("⚙️ Model: CNN (TensorFlow)")
st.sidebar.write("📊 Classes:", len(class_names))

# Main Title
st.markdown(
    "<h1 style='text-align: center; color: green;'>🌿 Plant Disease Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown("### 📸 Upload a plant leaf image")

file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if file:
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing image..."):
            processed = preprocess(img)
            prediction = model.predict(processed)

            index = np.argmax(prediction)
            confidence = prediction[0][index]
            label = class_names[index]

        st.success("✅ Prediction Complete")

        st.markdown(f"### 🦠 Disease: **{label}**")

        # Progress bar for confidence
        st.markdown("### 📊 Confidence")
        st.progress(float(confidence))

        st.write(f"{round(confidence * 100, 2)}%")

    st.markdown("---")

    # Info sections
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 📖 Description")
        st.info(disease_info[label]["description"])

    with col4:
        st.markdown("### 💊 Treatment")
        st.success(disease_info[label]["treatment"])