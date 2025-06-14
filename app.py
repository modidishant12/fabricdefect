import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Fabriconator", layout="centered")

from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
from keras.layers import DepthwiseConv2D

# 📛 Patch for DepthwiseConv2D to fix 'groups=1' error from Teachable Machine
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)

# ⏫ Disable scientific notation
np.set_printoptions(suppress=True)

# 🚀 Load model and labels with patch
@st.cache_resource
def load_model_and_labels():
    model = load_model("keras_Model.h5", custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D}, compile=False)
    labels = [label.strip() for label in open("labels.txt", "r").readlines()]
    return model, labels

model, class_names = load_model_and_labels()

# 🧠 Prediction function
def predict_teachable_model(image):
    # Teachable Machine expects 224x224 images
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Normalize image: (pixel / 127.5) - 1
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1

    # Reshape for model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = float(prediction[0][index]) * 100

    return class_name, confidence

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

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Fabric Image", use_container_width=True)

    with st.spinner("🔍 Analyzing..."):
        label, confidence = predict_teachable_model(image)

    st.success("✅ Detection Complete!")
    st.markdown(f"### 🎯 Result: **{label.upper()}**")
    st.markdown(f"**📊 Confidence:** {confidence:.2f}%")
