import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy as sp
from tensorflow.keras.models import load_model
import sqlite3
import os
from datetime import datetime
import uuid

st.set_page_config(
    page_title="Wildfire Prediction from Satellite Imagery",
    page_icon="🔥",
    layout="wide",
)

IM_SIZE = 224
IMAGE_RESIZE = (IM_SIZE, IM_SIZE, 3)
NUM_CLASSES = 2
CLASS_NAMES = ["nowildfire", "wildfire"]

# Directory setup
UPLOAD_DIR = "uploaded_images"
CAM_DIR = "cam_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CAM_DIR, exist_ok=True)


def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions
        (id TEXT PRIMARY KEY,
         image_path TEXT,
         no_wildfire_prob REAL,
         wildfire_prob REAL,
         cam_image_path TEXT,
         timestamp DATETIME)
    """
    )
    conn.commit()
    conn.close()


def save_prediction(image_path, no_wildfire_prob, wildfire_prob, cam_image_path):
    """Save prediction to database"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()
    pred_id = str(uuid.uuid4())
    c.execute(
        """
        INSERT INTO predictions
        (id, image_path, no_wildfire_prob, wildfire_prob, cam_image_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            pred_id,
            image_path,
            no_wildfire_prob,
            wildfire_prob,
            cam_image_path,
            datetime.now(),
        ),
    )
    conn.commit()
    conn.close()


def get_predictions():
    """Retrieve all predictions from database"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    predictions = c.fetchall()
    conn.close()
    return predictions


def delete_prediction(pred_id):
    """Delete a prediction from database and associated files"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()

    # Get file paths before deletion
    c.execute(
        "SELECT image_path, cam_image_path FROM predictions WHERE id = ?", (pred_id,)
    )
    result = c.fetchone()
    if result:
        image_path, cam_image_path = result

        # Delete files if they exist
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(cam_image_path):
            os.remove(cam_image_path)

    # Delete database entry
    c.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()


@st.cache_data
def load_models():
    """Load the trained models"""
    custom_model = load_model("saved_model/custom_model.keras")
    cam_model = load_model("saved_model/cam_model.keras")
    return custom_model, cam_model


def save_cam_plot(image_value, features, results, gap_weights):
    """Generate and save Class Activation Map"""
    features_for_img = features[0]
    prediction = results[0]

    class_activation_weights = gap_weights[:, 0]
    class_activation_features = sp.ndimage.zoom(
        features_for_img, (IM_SIZE / 10, IM_SIZE / 10, 1), order=2
    )
    cam_output = np.dot(class_activation_features, class_activation_weights)

    fig, ax = plt.subplots(figsize=(12, 12))
    cam_image = ax.imshow(cam_output, cmap="jet", alpha=0.5)
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
    plt.colorbar(cam_image)

    # Save the plot
    cam_filename = f"cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cam_path = os.path.join(CAM_DIR, cam_filename)
    plt.savefig(cam_path)
    plt.close()

    return cam_path


def process_image(image):
    """Process uploaded image for prediction"""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IM_SIZE, IM_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return np.expand_dims(img, axis=0)


def show_predictions_table():
    """Display predictions table with delete buttons"""
    predictions = get_predictions()

    # Add "Make Prediction" button at the top
    if st.button("Make Prediction", key="make_pred_btn"):
        st.session_state.show_predictions = False
        st.rerun()

    if not predictions:
        st.info("No predictions found in the database.")
        return

    for pred in predictions:
        pred_id, image_path, no_wildfire_prob, wildfire_prob, cam_path, timestamp = pred

        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 1])

        with col1:
            if os.path.exists(image_path):
                st.image(image_path, width=200)
            else:
                st.write("Image not found")

        with col2:
            st.write(f"No Wildfire: {no_wildfire_prob:.2f}%")

        with col3:
            st.write(f"Wildfire: {wildfire_prob:.2f}%")

        with col4:
            if os.path.exists(cam_path):
                st.image(cam_path, width=200)
            else:
                st.write("CAM image not found")

        with col5:
            if st.button("Delete", key=f"del_{pred_id}"):
                delete_prediction(pred_id)
                st.rerun()

        st.divider()


def main():
    # Initialize database
    init_db()

    # Initialize session state for page management
    if "show_predictions" not in st.session_state:
        st.session_state.show_predictions = False

    st.title("🔥 FireSight")
    st.markdown(
        """
    This application uses Convolutional Neural Networks (CNNs) to predict wildfires from satellite imagery.
    Upload a satellite image to get predictions and visualize the areas of interest using Class Activation Maps.
    """
    )

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

        # Add View Predictions button in sidebar
        if st.button("View Predictions History"):
            st.session_state.show_predictions = True
            st.rerun()

    # Show either predictions history or main prediction interface
    if st.session_state.show_predictions:
        st.subheader("Prediction History")
        show_predictions_table()
        return

    # Load models
    try:
        custom_model, cam_model = load_models()
        gap_weights = custom_model.layers[-1].get_weights()[0]
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    uploaded_file = st.file_uploader(
        "Upload a satellite image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Save uploaded image
        image = Image.open(uploaded_file)
        image_filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image_path = os.path.join(UPLOAD_DIR, image_filename)
        image.save(image_path)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        try:
            processed_image = process_image(image)

            features, results = cam_model.predict(processed_image)
            custom_results = custom_model.predict(processed_image)

            with col2:
                st.subheader("Prediction Results")
                no_wildfire_prob = custom_results[0][0] * 100
                wildfire_prob = custom_results[0][1] * 100

                st.write("No Wildfire Probability:", f"{no_wildfire_prob:.2f}%")
                st.write("Wildfire Probability:", f"{wildfire_prob:.2f}%")

                prediction = (
                    "Wildfire Detected! 🔥"
                    if wildfire_prob > 50
                    else "No Wildfire Detected ✅"
                )
                st.markdown(f"### Prediction: {prediction}")

            st.subheader("Class Activation Map")
            cam_path = save_cam_plot(processed_image, features, results, gap_weights)
            st.image(cam_path)

            # Save prediction to database
            save_prediction(image_path, no_wildfire_prob, wildfire_prob, cam_path)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    else:
        st.info("Please upload an image to get predictions")


if __name__ == "__main__":
    main()
