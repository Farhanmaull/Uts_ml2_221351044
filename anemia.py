import streamlit as st
import tensorflow  as tf # type: ignore
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="anaemia-prediction.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Anaemia Prediction")
st.write("Mendiagnosa Penyakit Anaemia")

# Form input pengguna
red = st.number_input("Red Pixel ", min_value=30.0, max_value=55.0, value=22.0)
green = st.number_input("Green Pixel ", min_value=20.0, max_value=35.0, value=30.0)
blue = st.number_input("Blue Pixel ", min_value=15.0, max_value=30.0, value=20.0)
hb = st.number_input("Hemoglobin ", min_value=0.0, max_value=20.0, value=10.0)

if st.button("Hasil Diagnosa"):
    # Preprocessing input
    input_data = np.array([[red,green,blue,hb]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    crop_name = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Hasil Diagnosa: **{crop_name.upper()}**")
