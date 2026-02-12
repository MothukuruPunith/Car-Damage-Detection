import os
import streamlit as st
import gdown
from model_helper import predict

# ---------------------------
# Model Auto Download
# ---------------------------

MODEL_PATH = "saved_model.pth"
FILE_ID = "1E9RF_cCUKvMzUIOAVKy6vmOIp1-GDMlC"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... Please wait ⏳")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully ✅")

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🚗 Car Damage Detection")
st.write("Upload a car image to detect damage type")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image_path = "temp_file.jpg"

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    prediction = predict(image_path)

    st.success(f"🔍 Predicted Class: {prediction}")
