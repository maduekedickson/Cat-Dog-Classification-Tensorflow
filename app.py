import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('cats_vs_dogs_model.h5')

st.title('Cats vs. Dogs Classifier')

# Upload an image
uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        st.write('This is a Dog!')
    else:
        st.write('This is a Cat!')
