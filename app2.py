import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
from PIL import Image

# Load the saved ANN model
model = tf.keras.models.load_model('model.h5')

# Define a function to preprocess inputs
def preprocess_input(inputs):
    # One-hot encode Geography
    geography_map = {"France": [1, 0, 0], "Germany": [0, 1, 0], "Spain": [0, 0, 1]}
    geography_encoded = geography_map.get(inputs['Geography'], [0, 0, 0])

    # Label encode Gender
    gender_encoded = 1 if inputs['Gender'] == "Male" else 0

    # Create feature array
    features = [
        inputs['CreditScore'],
        *geography_encoded,
        gender_encoded,
        inputs['Age'],
        inputs['Tenure'],
        inputs['Balance'],
        inputs['NumOfProducts'],
        inputs['HasCrCard'],
        inputs['IsActiveMember'],
        inputs['EstimatedSalary']
    ]

    # Scale the features (using a previously saved scaler)
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    scaled_features = scaler.transform([features])
    return scaled_features

# Add a background image
def add_background(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_path});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image (ensure the image file is in the same directory or provide a valid URL)
add_background(r"E:\Data_Science_cource\F__[DL_13Nov]\1__My_DLprojets_&_Code\D2_chun_pred_ANN\back.jpeg")

# Streamlit App
st.title("Customer Churn Prediction")
st.write("Predict whether a customer will churn based on their details.")

# Input fields in the sidebar
st.sidebar.header("Customer Details")
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30, step=1)
tenure = st.sidebar.slider("Tenure (Years with bank)", min_value=0, max_value=10, value=5, step=1)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=10000.0, step=100.0, format="%.2f")
num_of_products = st.sidebar.slider("Number of Products", min_value=1, max_value=4, value=2, step=1)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

# Prepare inputs
input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
    "EstimatedSalary": estimated_salary
}

# Prediction button
if st.button("Predict Churn"):
    try:
        # Preprocess inputs
        processed_input = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(processed_input)
        churn_probability = prediction[0][0]

        # Display results
        if churn_probability > 0.5:
            st.error(f"This customer is likely to churn. (Probability: {churn_probability:.2f})")
        else:
            st.success(f"This customer is unlikely to churn. (Probability: {churn_probability:.2f})")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
