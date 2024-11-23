import hashlib
import os
import sqlite3
import uuid
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

from algo import search_predictions, sort_predictions

st.set_page_config(
    page_title="Wildfire Prediction from Satellite Imagery",
    page_icon="🔥",
    layout="wide",
)

IM_SIZE = 224
IMAGE_RESIZE = (IM_SIZE, IM_SIZE, 3)
NUM_CLASSES = 2
CLASS_NAMES = ["nowildfire", "wildfire"]

UPLOAD_DIR = "uploaded_images"
CAM_DIR = "cam_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CAM_DIR, exist_ok=True)


def init_auth_db():
    """Initialize authentication database"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users
        (id TEXT PRIMARY KEY,
         username TEXT UNIQUE,
         password_hash TEXT,
         created_at DATETIME)
    """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions
        (id TEXT PRIMARY KEY,
         user_id TEXT,
         image_path TEXT,
         no_wildfire_prob REAL,
         wildfire_prob REAL,
         cam_image_path TEXT,
         timestamp DATETIME,
         FOREIGN KEY (user_id) REFERENCES users(id))
    """
    )
    conn.commit()
    conn.close()


def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password):
    """Create new user in database"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()
    try:
        user_id = str(uuid.uuid4())
        c.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, hash_password(password), datetime.now()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    """Verify user credentials"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    if result and result[1] == hash_password(password):
        return result[0]
    return None


def show_login_page():
    """Display login/signup interface"""
    st.title("🔥 FireSight - Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                user_id = verify_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    if create_user(new_username, new_password):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Username already exists")


def save_prediction(
    user_id, image_path, no_wildfire_prob, wildfire_prob, cam_image_path
):
    """Save prediction to database"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()
    pred_id = str(uuid.uuid4())
    c.execute(
        """
        INSERT INTO predictions
        (id, user_id, image_path, no_wildfire_prob, wildfire_prob, cam_image_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            pred_id,
            user_id,
            image_path,
            no_wildfire_prob,
            wildfire_prob,
            cam_image_path,
            datetime.now(),
        ),
    )
    conn.commit()
    conn.close()


def get_predictions(user_id):
    """Retrieve predictions for specific user"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()
    c.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,),
    )
    predictions = c.fetchall()
    conn.close()
    return predictions


def delete_prediction(pred_id):
    """Delete a prediction from database and associated files"""
    conn = sqlite3.connect("wildfire_predictions.db")
    c = conn.cursor()

    c.execute(
        "SELECT image_path, cam_image_path FROM predictions WHERE id = ?", (pred_id,)
    )
    result = c.fetchone()
    if result:
        image_path, cam_image_path = result

        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(cam_image_path):
            os.remove(cam_image_path)

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
    """
    Enhanced predictions table with sorting and searching capabilities
    """
    predictions = get_predictions(st.session_state.user_id)

    if st.button("Make New Prediction", key="make_pred_btn"):
        st.session_state.show_predictions = False
        st.rerun()

    if not predictions:
        st.info("No predictions found in the database.")
        return

    # Sort and search controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["date", "wildfire_prob", "no_wildfire_prob"],
            key="sort_by",
        )

    with col2:
        search_term = st.text_input(
            "Search (date: YYYY-MM-DD or probability: >50, <25)",
            key="search_term",
            help="Enter a date (YYYY-MM-DD) or probability threshold (>50 or <25)",
        )

    with col3:
        sort_order = st.selectbox(
            "Order", options=["Descending", "Ascending"], key="sort_order"
        )

    # Apply sorting and searching with performance metrics
    start_time = datetime.now()
    sorted_predictions = sort_predictions(
        predictions, sort_by, reverse=(sort_order == "Descending")
    )
    sort_time = (datetime.now() - start_time).total_seconds() * 1000

    if search_term:
        start_time = datetime.now()
        filtered_predictions = search_predictions(sorted_predictions, search_term)
        search_time = (datetime.now() - start_time).total_seconds() * 1000

        if not filtered_predictions:
            st.warning("No matching predictions found.")
            return
        sorted_predictions = filtered_predictions

        # Display performance metrics in an expander
        with st.expander("Performance Metrics"):
            st.write(f"Search completed in {search_time:.2f}ms")
            st.write(f"Sort completed in {sort_time:.2f}ms")

    st.write(f"Showing {len(sorted_predictions)} predictions")

    # Display predictions in a modern layout
    for pred in sorted_predictions:
        (
            pred_id,
            user_id,
            image_path,
            no_wildfire_prob,
            wildfire_prob,
            cam_path,
            timestamp,
        ) = pred

        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 1])

            with col1:
                if os.path.exists(image_path):
                    st.image(image_path, width=200)
                else:
                    st.write("Image not found")

            with col2:
                st.metric("No Wildfire", f"{no_wildfire_prob:.1f}%")

            with col3:
                st.metric("Wildfire", f"{wildfire_prob:.1f}%")

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
    init_auth_db()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "show_predictions" not in st.session_state:
        st.session_state.show_predictions = False

    if not st.session_state.authenticated:
        show_login_page()
        return

    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()

    st.title(f"🔥 FireSight - Welcome, {st.session_state.username}!")
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

        if st.button("View Predictions History"):
            st.session_state.show_predictions = True
            st.rerun()

    if st.session_state.show_predictions:
        st.subheader(f"Prediction History for {st.session_state.username}")
        show_predictions_table()
        return

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

            save_prediction(
                st.session_state.user_id,
                image_path,
                no_wildfire_prob,
                wildfire_prob,
                cam_path,
            )

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    else:
        st.info("Please upload an image to get predictions")


if __name__ == "__main__":
    main()
