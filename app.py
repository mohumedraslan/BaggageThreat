import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model = YOLO("best.pt")

# Streamlit app
st.title("Baggage Threat Detection")
st.write("Upload an image to detect potential threats using YOLOv8.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference
    results = model(img_cv2)
    output_img = results[0].plot()  # Draw bounding boxes

    # Convert back to RGB for display
    output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # Display result
    st.image(output_img_rgb, caption="Detection Result", use_column_width=True)