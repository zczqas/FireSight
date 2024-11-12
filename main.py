import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import scipy as sp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import TFSMLayer
import io
import os

# Set page config
st.set_page_config(
    page_title="Wildfire Prediction from Satellite Imagery",
    page_icon="🔥",
    layout="wide",
)

# Constants
IM_SIZE = 224
IMAGE_RESIZE = (IM_SIZE, IM_SIZE, 3)
NUM_CLASSES = 2
CLASS_NAMES = ["nowildfire", "wildfire"]


@st.cache_data
def load_models():
    """Load the trained models"""
    custom_model = tf.keras.Sequential(
        [TFSMLayer("saved_model/custom_model", call_endpoint="serving_default")]
    )
    cam_model = tf.keras.Sequential(
        [TFSMLayer("saved_model/cam_model", call_endpoint="serving_default")]
    )
    return custom_model, cam_model


def show_cam(image_value, features, results, gap_weights):
    """Generate and display Class Activation Map"""
    features_for_img = features[0]
    prediction = results[0]

    class_activation_weights = gap_weights[:, 0]
    class_activation_features = sp.ndimage.zoom(
        features_for_img, (IM_SIZE / 10, IM_SIZE / 10, 1), order=2
    )
    cam_output = np.dot(class_activation_features, class_activation_weights)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(cam_output, cmap="jet", alpha=0.5)
    ax.imshow(tf.squeeze(image_value), alpha=0.5)
    ax.set_title("Class Activation Map")
    plt.figtext(
        0.5,
        0.05,
        f"No Wildfire Probability: {results[0][0] * 100:.2f}%\n"
        f"Wildfire Probability: {results[0][1] * 100:.2f}%",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "green", "alpha": 0.5, "pad": 3},
    )
    plt.colorbar()
    st.pyplot(fig)


def process_image(image):
    """Process uploaded image for prediction"""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IM_SIZE, IM_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return np.expand_dims(img, axis=0)


def main():
    # Title and description
    st.title("🔥 Wildfire Prediction from Satellite Imagery")
    st.markdown(
        """
    This application uses Convolutional Neural Networks (CNNs) to predict wildfires from satellite imagery.
    Upload a satellite image to get predictions and visualize the areas of interest using Class Activation Maps.
    """
    )

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            """
        This model was trained on satellite images with a resolution of 350x350 pixels,
        categorized into two classes: Wildfire and No Wildfire.
        
        The dataset contains satellite imagery from various locations,
        capturing different environmental conditions that may indicate wildfire risk.
        """
        )

        st.header("Model Information")
        st.write("Image Size:", IM_SIZE)
        st.write("Number of Classes:", NUM_CLASSES)
        st.write("Classes:", ", ".join(CLASS_NAMES))

    # Load models
    try:
        custom_model, cam_model = load_models()
        gap_weights = custom_model.layers[-1].get_weights()[0]
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a satellite image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        # Process image and make predictions
        try:
            processed_image = process_image(image)

            # Get predictions from both models
            features, results = cam_model.predict(processed_image)
            custom_results = custom_model.predict(processed_image)

            # Display results
            with col2:
                st.subheader("Prediction Results")
                st.write(
                    "No Wildfire Probability:", f"{custom_results[0][0] * 100:.2f}%"
                )
                st.write("Wildfire Probability:", f"{custom_results[0][1] * 100:.2f}%")

                prediction = (
                    "Wildfire Detected! 🔥"
                    if custom_results[0][1] > 0.5
                    else "No Wildfire Detected ✅"
                )
                st.markdown(f"### Prediction: {prediction}")

            # Show Class Activation Map
            st.subheader("Class Activation Map")
            show_cam(processed_image, features, results, gap_weights)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    else:
        st.info("Please upload an image to get predictions")


if __name__ == "__main__":
    main()
