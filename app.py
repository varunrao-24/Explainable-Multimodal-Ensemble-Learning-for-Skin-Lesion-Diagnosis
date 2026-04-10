import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Skin Lesion App")

st.title("🧬 Skin Lesion Classification System")
st.warning(
    "⚠️ **Medical Disclaimer**\n\n"
    "This application is intended for **educational and research purposes only**. "
    "It is **not a substitute for professional medical advice, diagnosis, or treatment**. "
    "Always consult a qualified dermatologist for clinical decisions."
)
st.write("Upload a skin lesion image to get prediction.")

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(
            "models/effnetb3.keras",
            compile=False
        )
    except Exception as e:
        st.error("Model loading failed")
        st.exception(e)
        st.stop()

st.info("Loading model...")
model = load_model()
st.success("Model loaded successfully ✅")

CLASS_NAMES = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
def preprocess_image(uploaded_file, img_size=(300, 300)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(img_size)

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT: same preprocessing as training
    img_array = preprocess_input(img_array)

    return img_array

uploaded_file = st.file_uploader(
    "Upload skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Skin Lesion Image", width=300)

    img_array = preprocess_image(uploaded_file)

    with st.spinner("🔍 Analyzing image..."):
        preds = model.predict(img_array)

    probs = preds[0]
    pred_idx = np.argmax(probs)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    st.subheader("🧠 Prediction Result")
    st.success(f"**Predicted Class:** {pred_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Risk message (keep your logic ✔)
    if pred_class in ["MEL", "SCC"]:
        st.error(
        "🚨 **Warning:** High-risk lesion detected. "
        "Please consult a dermatologist immediately.")
        
    else:
        st.success("✅ Low-risk lesion detected.")

    # Top-3 predictions (new, very important)
    st.subheader("📊 Top Predictions")
    top3_idx = probs.argsort()[-3:][::-1]

    for i in top3_idx:
        st.write(f"{CLASS_NAMES[i]} : {probs[i]*100:.2f}%")




