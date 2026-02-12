import streamlit as st
import torch
from torchvision import models,transforms
from torch import nn
from model_helper import predict
st.title("Car damage detection")

uploaded_file = st.file_uploader("Upload the file",type=["jpg","png"])

if uploaded_file:
    image_path =  "temp_file.jpg"
    with open(image_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file,caption='Uploaded File',use_container_width=True)
        prediction = predict(image_path)
        st.info(f"Predicted Class : {prediction}")
    