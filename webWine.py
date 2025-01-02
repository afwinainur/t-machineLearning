import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Judul aplikasi
st.title("Wine Quality Prediction")

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Input pengguna
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0 )
    citric_acid = st.number_input("Citric Acid", min_value=0.0 )
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0 )
    chlorides = st.number_input("Chlorides", min_value=0.0 )

with col2:
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0 )
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0 )
    density = st.number_input("Density", min_value=0.0)
    pH = st.number_input("pH", min_value=0.0 )
    sulphates = st.number_input("Sulphates", min_value=0.0 )
    alcohol = st.number_input("Alcohol", min_value=0.0 )

# Tambahkan tombol untuk prediksi atau tindakan lainnya
if st.button("Prediksi"):
    # Pastikan input berbentuk array 2D
    input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                   free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
    
    # Lakukan prediksi
    prediksi = model.predict(input_data)
    
    # Tampilkan hasil prediksi
    st.write(f"Prediksi: {prediksi[0]}")