import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<h1 style='text-align:center; color:#1B3A6B;'>🧠 Brain Tumor MRI Classifier</h1>
<p style='text-align:center; color:gray;'>Upload a brain MRI scan and get an instant AI-powered classification</p>
<hr>
""", unsafe_allow_html=True)

CLASSES  = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
IMG_SIZE = 128

@st.cache_resource
def load_model():
    if os.path.exists("brain_tumor_model.h5"):
        return tf.keras.models.load_model("brain_tumor_model.h5")
    else:
        return None

model = load_model()

if model is None:
    st.error("Model file not found. Please upload brain_tumor_model.h5")
    st.stop()

st.subheader("Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose a brain MRI image (.jpg or .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Prediction Result")

        img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions     = model.predict(img_array, verbose=0)[0]
        predicted_class = CLASSES[np.argmax(predictions)]
        confidence      = np.max(predictions) * 100

        color = "#e74c3c" if predicted_class != "No Tumor" else "#27ae60"

        st.markdown(f"""
        <div style='background-color:{color}22; border-left: 5px solid {color};
                    padding: 20px; border-radius: 8px; margin-top: 20px;'>
            <h2 style='color:{color}; margin:0;'>{predicted_class}</h2>
            <p style='color:gray; margin:5px 0 0 0;'>Confidence: <b>{confidence:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("All Class Probabilities")
        for cls, prob in zip(CLASSES, predictions):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

st.markdown("---") 
with st.expander("About this App"):
    st.markdown("""
    This app uses a MobileNetV2 deep learning model trained on the
    Brain Tumor MRI Dataset from Kaggle to classify brain MRI scans into 4 categories:
    - Glioma
    - Meningioma
    - Pituitary Tumor
    - No Tumor

    **Disclaimer:** This tool is for academic demonstration only.
    Do NOT use it for actual medical diagnosis.
    """)

st.markdown(
    "<p style='text-align:center; color:lightgray; font-size:12px;'>"
    "Final Year Project — AI Assignment | Brain Tumor Detection</p>",
    unsafe_allow_html=True
)