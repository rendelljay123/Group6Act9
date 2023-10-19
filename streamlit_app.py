import numpy as np
import streamlit as st
import tensorflow as tf
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
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_custom_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    image_reshape = np.reshape(image, (1, 256, 256, 3))
    prediction = model.predict(image_reshape)
    return prediction


# Define model paths for different plant types
model_paths = {
    "tomato": "./model/Tomato_Model.h5",
    "cotton": "./model/Cotton_Model.h5",
    "potato": "./model/Potato_Model.h5",
}

# Load default model for initialization
default_model_path = "./model/Tomato_Model.h5"
model = load_custom_model(default_model_path)

# Define class labels for plant diseases
class_names = {
    "tomato": {0: "Tomato_Early_blight", 1: "Tomato_Leaf_Mold", 2: "Tomato_healthy"},
    "cotton": {
        0: "diseased cotton leaf", 
        1: "diseased cotton plant", 
        2: "fresh cotton leaf", 
        3: "fresh cotton plant",
    },
    "potato": {
        0: "Potato___Early_blight", 
        1: "Potato___Late_blight", 
        2: "Potato___healthy",
    },
}

# Define disease information for each plant type
disease_info = {
    'tomato': {
        'Tomato_Early_blight': "Early blight is a common fungal disease that affects tomato plants. Early Blight typically starts as small, dark brown lesions with concentric rings on lower leaves...",
        'Tomato_Leaf_Mold': "Tomato leaf mold is a foliar disease that primarily affects the leaves. Management of Leaf Mold involves maintaining proper air circulation and reducing humidity levels in greenhouses...",
        'Tomato_healthy': "Your tomato plant looks healthy!",
    },
    'cotton': {
    'diseased cotton leaf': "Diseased cotton leaves often show symptoms like yellowing and wilting, which can be caused by various pathogens like Fusarium and Verticillium wilt, leading to a decline in cotton plant health and yield. Regular monitoring of cotton fields for early signs of leaf diseases is crucial for effective disease management.",
    'diseased cotton plant': "Cotton plant diseases can lead to reduced yield and fiber quality, affecting the overall cotton crop production. These diseases can be exacerbated by environmental conditions such as high humidity and insufficient air circulation. Implementing proper crop rotation, pest control, and maintaining healthy soil can help mitigate the risks associated with cotton plant diseases.",
    'fresh cotton leaf': "A healthy cotton leaf is characterized by vibrant green color, turgid structure, and absence of discoloration or abnormalities. Regular monitoring and maintaining optimal growing conditions contribute to the overall health of cotton leaves. Your cotton leaf appears healthy!",
    'fresh cotton plant': "A healthy cotton plant exhibits vigorous growth, with well-formed and vibrant leaves. The absence of wilting, discoloration, or stunted growth is indicative of a thriving cotton plant. Adequate water, nutrient balance, and pest control contribute to the overall health of your cotton plant. Your cotton plant appears healthy!",
    },
    'potato': {
        'Potato___Early_blight': "Early blight in potatoes can cause dark lesions on leaves. It is primarily caused by the fungus Alternaria solani. The symptoms include small, dark spots with concentric rings that may merge as the disease progresses. ",
        'Potato___Late_blight': "Late blight is a serious disease of potato and tomato crops. Late blight symptoms include water-soaked lesions on leaves, which can rapidly enlarge and become necrotic. Infected plants may exhibit a foul odor.",
        'Potato___healthy': "Your potato plant looks healthy! Maintain your potato plant's health by adhering to good agricultural practices, which involve proper irrigation, fertilization, and vigilant monitoring for signs of pests or diseases. Regularly inspect the foliage, apply organic fungicides.",
    }
}


st.write("# Plant Disease Detection")
st.write("### CPE 028 - DevOps")
st.write("Baltazar, Rendell Jay; Dalmacio, Andre Christian; Makiramdam, Releigh; Orbeta, John Mark; Co, Jericho")

# Allow the user to select the plant type
plant_type = st.selectbox("Select Plant Type", ("tomato", "cotton", "potato"))

# Allow the user to upload an image
file = st.file_uploader("Upload an image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    try:
        # Load the selected model based on the chosen plant type
        model_path = model_paths.get(plant_type)
        if model_path:
            model = load_custom_model(model_path)
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
                    string = f"The {plant_type} plant is {class_name}"
                    st.success(string)

                    # Display disease information in an expander
                    with st.expander("Learn More About the Disease"):
                        disease_description = disease_info.get(plant_type, {}).get(class_name, "No information available.")
                        st.write(disease_description)
                else:
                    st.text("Invalid class index")
            else:
                st.text("Invalid class labels for the selected plant type")
        else:
            st.text("Invalid file. Please upload a valid image file.")
    except Exception as e:
        st.text("Error occurred while processing the image.")
        st.text(str(e))
