import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from keras.layers import DepthwiseConv2D
import os

# ✅ Streamlit setup
st.set_page_config(page_title="Fabriconator", layout="centered")

# 🔧 Patch DepthwiseConv2D (for Teachable Machine models)
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)

# 🔁 Load model from repo path
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), "keras_model.h5")
    model = load_model(model_path, compile=False, custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D})
    return model

model = load_trained_model()

# 🔤 Hardcoded class names (based on training)
CLASS_NAMES = ["Hole", "spot", "Line", "Good"]  # Change order as per your model output

# 🔍 Prediction function
def predict(image: Image.Image):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    normalized = (img_array / 127.5) - 1
    data = np.expand_dims(normalized, axis=0)

    prediction = model.predict(data)
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index]) * 100
    label = CLASS_NAMES[index]

    return label, confidence

# ========== 🌐 Streamlit UI ==========
st.sidebar.title("🧵 Fabriconator")
st.sidebar.image("logo.jpg", caption="Fabriconator", use_container_width=True)
st.sidebar.markdown("Upload a fabric image to detect defects.")
uploaded_file = st.sidebar.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

st.title("🧵 Fabriconator - Fabric Defect Detector")
st.markdown("""
Fabriconator is an AI-powered tool trained to detect defects in fabric images.  
It classifies fabric into one of the following:
- 🕳️ **Hole**
- ⚫ **Spot**
- ➖ **Line**
- ✅ **Good**

Upload a fabric image from the sidebar to get started.
""")

# 🖼️ Show image and prediction
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Fabric Image", use_column_width=True)

    with st.spinner("🔍 Analyzing fabric..."):
        label, confidence = predict(image)

    st.success("✅ Analysis Complete!")
    st.markdown(f"### 🎯 Predicted Class: **{label.upper()}**")
    st.markdown(f"**📊 Confidence Score:** `{confidence:.2f}%`")
