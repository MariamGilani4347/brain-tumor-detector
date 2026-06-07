import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="centered")

st.markdown("""
<h1 style='text-align:center; color:#1B3A6B;'>🧠 Brain Tumor MRI Classifier</h1>
<p style='text-align:center; color:gray;'>Upload a brain MRI scan and get an instant AI-powered classification</p>
<hr>
""", unsafe_allow_html=True)

CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

@st.cache_resource
def load_model():
    return ort.InferenceSession("brain_tumor_model.onnx")

def predict(image):
    session = load_model()
    img_array = np.array(image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})
    return output[0][0]

st.subheader("Upload MRI Image")
uploaded_file = st.file_uploader("Choose a brain MRI image (.jpg or .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Prediction Result")
        predictions = predict(image)
        predicted_class = CLASSES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        color = "#e74c3c" if predicted_class != "No Tumor" else "#27ae60"
        st.markdown(f"""
        <div style='background-color:{color}22; border-left:5px solid {color};
                    padding:20px; border-radius:8px; margin-top:20px;'>
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
    This app uses a MobileNetV2 model trained on the Brain Tumor MRI Dataset from Kaggle.
    - Glioma | Meningioma | Pituitary Tumor | No Tumor
    
    **Disclaimer:** For academic use only. Not for medical diagnosis.
    """)
