import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load the YOLO model
model = YOLO("Model/best.pt")  # Replace with your custom-trained model path

# Streamlit app layout
st.set_page_config(layout="wide")
st.title("Brain Tumor Segmentation with YOLOv11")

# Create columns for input and output images
col1, col2 = st.columns(2)

with col1:
    st.header("Input Image")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

with col2:
    st.header("Output Image")
    st.subheader("Segmented Output\n")
    output_placeholder = st.empty()
    # Prediction display at the bottom

prediction_placeholder = st.success("")

# Process the uploaded file
if uploaded_file is not None:
    # Load the input image
    input_image = Image.open(uploaded_file)
    col1.image(input_image, caption="Uploaded Image", use_column_width=False)

    # Convert the image to a format YOLO model can process
    image_np = np.array(input_image.convert("RGB"))

    # Perform prediction
    results = model.predict(source=image_np, save=False, conf=0.25)  # Perform inference
    
    # Extract prediction details
    segmented_image = results[0].plot()  # Get the segmented image
    predicted_classes = [model.names[int(box.cls)] for box in results[0].boxes]

    # Display segmented output
    col2.image(segmented_image, caption="Segmented Output", use_column_width=True)

    # Display predicted classes
    predicted_class_text = ", ".join(predicted_classes)
    prediction_placeholder.success(f"### Predicted Class: {predicted_class_text}")

else:
    prediction_placeholder.success("### Upload an image to see results.")