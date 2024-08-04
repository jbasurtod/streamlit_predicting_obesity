import streamlit as st
import pickle
import pandas as pd

# Set the page configuration
st.set_page_config(page_title="Obesity Risk Prediction", layout="wide")

# Sidebar with a link
with st.sidebar:
    st.markdown("<h2>Created by Juan C Basurto</h2>", unsafe_allow_html=True)
    st.markdown("[GitHub Profile](https://github.com/jbasurtod)", unsafe_allow_html=True)

# Title of the Dashboard
st.title("Obesity Risk Prediction")

# Brief description of the model
st.markdown("""
    This application uses a Random Forest model trained to predict obesity based on five criteria. The model was trained using the **Estimation of Obesity Levels Based On Eating Habits and Physical Condition** dataset, which can be found [here](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). The model's recall in identifying obesity is 0.87, indicating a high ability to correctly identify individuals who fall into the obesity categories (I, II, or III). More information about the model training can be found in [this Kaggle Notebook](https://www.kaggle.com/code/jbasurtod/predicting-obesity-with-random-forests).
""")

# Load the model from the pickle file
model_path = "models/rf_model.pkl"  # Update with the correct file name
try:
    with open(model_path, 'rb') as model_file:
        model, threshold = pickle.load(model_file)
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Requesting variables and their input types with default values

# Age - Integer
Age = st.number_input("What is your age? (Age)", min_value=0, max_value=120, value=31, step=1)

# family_history_with_overweight_1 - Yes or No (default: No)
family_history_with_overweight_1 = st.selectbox(
    "Do you have a family history with overweight? (family_history_with_overweight_1)",
    options=["Yes", "No"],
    index=0  # Default to "No"
)

# FAVC_1 - Yes or No (default: No)
FAVC_1 = st.selectbox(
    "Do you frequently consume fruits and vegetables? (FAVC_1)",
    options=["Yes", "No"],
    index=0  # Default to "No"
)

# CAEC_Sometimes - Yes or No (default: No)
CAEC_Sometimes = st.selectbox(
    "Do you sometimes consume high caloric food? (CAEC_Sometimes)",
    options=["Yes", "No"],
    index=0  # Default to "No"
)

# SCC_1 - Yes or No (default: No)
SCC_1 = st.selectbox(
    "Do you monitor the calories you eat daily? (SCC_1)",
    options=["Yes", "No"],
    index=1  # Default to "No"
)

# Convert Yes/No to 1/0
def convert_yes_no(value):
    return 1 if value == "Yes" else 0

# Button to predict and show results
if st.button("Predict"):
    # Prepare the data for prediction
    input_data = pd.DataFrame({
        "age": [Age],
        "family_history_with_overweight": [convert_yes_no(family_history_with_overweight_1)],
        "FAVC": [convert_yes_no(FAVC_1)],
        "CAEC": [convert_yes_no(CAEC_Sometimes)],
        "SCC": [convert_yes_no(SCC_1)]
    })
    
    try:
        # Make prediction
        #prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of obesity class

        # Get the prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        # Determine the final prediction based on the threshold
        prediction = (probabilities[1] >= threshold).astype(int)

        # Determine the result message and color
        if prediction == 1:
            result_message = "Obesity Classification Detected"
            color = "#FFA500"  # Orange color
        else:
            result_message = "No Obesity Classification Detected"
            color = "#1E90FF"  # Blue color

        # Convert probability to percentage
        probability_percentage = f"{probability * 100:.2f}%"

        # Display the result with the appropriate color and probability
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{result_message}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Probability: {probability_percentage}</h3>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
