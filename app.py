import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# ----------------------------------------------------------------
# 🔹 Download model from Google Drive (only if not already present)
# ----------------------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1zwc2nTpzS5V5qjsnIPAhekbpu6XocXWm"
MODEL_PATH = "trained_plant_disease_model.keras"

@st.cache_resource(show_spinner=False)
def load_model_file():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model... please wait!"):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model_file()

# ----------------------------------------------------------------
# 🔹 TensorFlow Model Prediction
# ----------------------------------------------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# ----------------------------------------------------------------
# 🔹 Sidebar
# ----------------------------------------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# ----------------------------------------------------------------
# 🔹 Home Page
# ----------------------------------------------------------------
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM 🌿")
    image_path = "home_page.jpeg"
    st.image(image_path, caption='Uploaded Image', use_container_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 🌱🔍  
    Upload a plant leaf image to identify diseases using our trained deep learning model.

    ### 🌼 How It Works
    1. **Upload Image** – Go to the **Disease Recognition** page.
    2. **Analysis** – Model predicts the most probable disease.
    3. **Results** – Get instant feedback on the plant’s condition.

    ### 💡 Why Choose This App?
    - **Accurate:** Uses deep learning for high precision.
    - **Fast:** Results within seconds.
    - **Simple UI:** Built with Streamlit for ease of use.
    """)

# ----------------------------------------------------------------
# 🔹 About Page
# ----------------------------------------------------------------
elif app_mode == "About":
    st.header("About Dataset & Project")
    st.markdown("""
    #### 📘 About Dataset
    - Dataset recreated using offline augmentation.
    - Original dataset from PlantVillage (public GitHub repo).
    - Contains **~87K RGB images** of healthy and diseased leaves across **38 classes**.
    - Split:
      - **Train:** 70,295 images  
      - **Validation:** 17,572 images  
      - **Test:** 33 images

    #### 🎯 Project Goal
    Build an efficient system to **automatically detect plant diseases** from leaf images to help farmers and researchers take timely actions.
    """)

# ----------------------------------------------------------------
# 🔹 Disease Recognition Page
# ----------------------------------------------------------------
elif app_mode == "Disease Recognition":
    st.header("🌾 Disease Recognition")
    test_image = st.file_uploader("Choose an Image of a Plant Leaf:", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, width=4, use_container_width=True)

    if st.button("🔍 Predict"):
        if test_image:
            st.snow()
            st.write("Analyzing image... please wait ⏳")
            result_index = model_prediction(test_image)

            # Reading Labels
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            st.success(f"🌱 Model Prediction: **{class_name[result_index]}** ✅")
        else:
            st.warning("⚠️ Please upload an image before clicking Predict.")
