import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model(r'C:\Users\ssjsm\OneDrive\Desktop\Datasets\cat and dog prediction\cat_and_dog_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make predictions
def predict(image_array):
    prediction = model.predict(image_array)
    return prediction

# Streamlit App
def main():
    # Add background image
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSspAGMlS6IzEDsi-ZOdwxURAvTyf76VYRKTGX11JNg16tZ0GxxJWsdGF51OoraU26TKYU&usqp=CAU');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Cat vs Dog Image Classifier")
    st.sidebar.title("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.sidebar.button("Predict", key="predict_button", on_click=predict_image, args=(uploaded_file,))

def predict_image(uploaded_file):
    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)

    # Make prediction
    prediction = predict(img_array)

    # Display the result
    if prediction[0][0] > 0.5:
        st.sidebar.markdown("<p style='font-size:20px; color:green;'>Prediction: Dog</p>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<p style='font-size:20px; color:red;'>Prediction: Cat</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
