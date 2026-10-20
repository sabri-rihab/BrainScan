import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BrainScan AI",
    page_icon="🧠",
    layout="centered"
)
# ── constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = str(Path(__file__).parent.parent / "models" / "best_model.keras")
IMG_SIZE    = (224, 224)
LABEL_MAP   = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
LABEL_COLOR = {
    "Glioma":      "#e74c3c",
    "Meningioma":  "#e67e22",
    "No Tumor":    "#2ecc71",
    "Pituitary":   "#3498db",
}

# ── load model once ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── prediction function ───────────────────────────────────────────────────────
def predict(image_bytes):
    # decode bytes → numpy array
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # same preprocessing as training
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # (1, 224, 224, 3)

    probs        = model.predict(img, verbose=0)[0]   # shape (4,)
    class_index  = int(np.argmax(probs))
    class_name   = LABEL_MAP[class_index]
    confidence   = float(probs[class_index])

    return class_name, confidence, probs

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 BrainScan AI")
st.write("Upload a brain MRI image to detect the type of tumor.")

st.divider()

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image_bytes = uploaded_file.read()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image_bytes, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        with st.spinner("Analyzing..."):
            class_name, confidence, probs = predict(image_bytes)

        color = LABEL_COLOR[class_name]
        st.markdown(
            f"<h2 style='color:{color}'>{class_name}</h2>",
            unsafe_allow_html=True
        )
        st.metric(label="Confidence", value=f"{confidence:.2%}")

        st.divider()

        # probability bar for each class
        st.write("**All class probabilities:**")
        for idx, prob in enumerate(probs):
            label = LABEL_MAP[idx]
            st.progress(float(prob), text=f"{label}: {prob:.2%}")
