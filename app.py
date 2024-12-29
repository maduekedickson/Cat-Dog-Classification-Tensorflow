import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('cats_vs_dogs_model.h5')

# Streamlit UI
st.image("images.jpeg")
st.title('Cats vs. Dogs Classifier')
st.write("Upload a JPG image, and let the AI classify it as a Cat or a Dog!")

# Upload an image
uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    
    if prediction[0][0] > 0.5:
        st.write(f'This is a Dog with {confidence * 100:.2f}% confidence!')
    else:
        st.write(f'This is a Cat with {confidence * 100:.2f}% confidence!')
