import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Set page configuration with background image and gradient
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Define CSS for background image and gradient
css = """
body {
    background-image: url('./assets/leaf-bg.jpg');
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-image: linear-gradient(rgba(0, 153, 51, 0.2), rgba(255, 255, 255, 0.2));
    background-size: cover; /* Adjust the percentage as desired */
    background-position: center;
}
"""

# Apply CSS to Streamlit
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    image_reshape = np.reshape(image, (1, 256, 256, 3))
    prediction = model.predict(image_reshape)
    return prediction

# Define model paths for different plant types
model_paths = {
    'tomato': './model/Tomato_Model.h5',
    'cotton': './model/Cotton_Model.h5',
    'potato': './model/Potato_Model.h5'
}

# Load default model for initialization
default_model_path = './model/Tomato_Model.h5'
model = load_model(default_model_path)

# Define class labels for plant diseases
class_names = {
    'tomato': {0: 'Tomato_Early_blight', 1: 'Tomato_Leaf_Mold', 2: 'Tomato_healthy'},
    'cotton': {0: 'diseased cotton leaf', 1: 'diseased cotton plant', 2: 'fresh cotton leaf', 3: 'fresh cotton plant'},
    'potato': {0: 'Potato___Early_blight', 1: 'Potato___Late_blight', 2: 'Potato___healthy'}
}


st.write("# Plant Disease Detection")
st.write("### CPE 019 - Emerging Technologies in CpE 3")
st.write("Baltazar, Rendell Jay; Jarabejo, Joshua; La Madrid, Angelo H.")

# Allow the user to select the plant type
plant_type = st.selectbox("Select Plant Type", ('tomato', 'cotton', 'potato'))

# Allow the user to upload an image
file = st.file_uploader("Upload an image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    try:
        # Load the selected model based on the chosen plant type
        model_path = model_paths.get(plant_type)
        if model_path:
            model = load_model(model_path)
        else:
            st.text("Invalid plant type")

        image = Image.open(file) if file else None
        if image:
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            class_labels = class_names.get(plant_type)
            if class_labels:
                class_index = np.argmax(prediction)
                class_name = class_labels.get(class_index)
                if class_name:
                    string = f"The {plant_type} plant is {class_name} "
                    st.success(string)
                else:
                    st.text("Invalid class index")
            else:
                st.text("Invalid class labels for the selected plant type")
        else:
            st.text("Invalid file. Please upload a valid image file.")
    except Exception as e:
        st.text("Error occurred while processing the image.")
        st.text(str(e))