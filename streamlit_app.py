import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
from PIL import Image

# Load your pre-trained model
model = load_model('Cat_VS_Dog.h5')  # Replace with your actual model path

st.title("Cat vs Dog!")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image_path = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image_path, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image to match the input format of the model
    img = image_path.resize((150, 150))  # Resize to the same size as used during model training
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Output the result
    if prediction[0] > 0.5:
        st.write("It's a Dog!")
    else:
        st.write("It's a Cat!")