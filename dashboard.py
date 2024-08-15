import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set the page configuration
st.set_page_config(page_title="Obesity Risk Prediction", layout="wide")

# Sidebar with a link
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: left;">
            <img src="https://avatars.githubusercontent.com/u/4228249?v=4" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h2>Created by Juan C Basurto</h2>", unsafe_allow_html=True)
    st.markdown("[GitHub Profile](https://github.com/jbasurtod)", unsafe_allow_html=True)

# Title of the Dashboard
st.title("Obesity Risk Prediction")

# Updated brief description of the models
st.markdown("""
    This application uses two models to predict obesity based on 5 criteria after being trained with SelectKBest. The [Random Forest model](https://www.kaggle.com/code/jbasurtod/predicting-obesity-with-random-forests#Testing-the-Random-Forest-Model) achieves a recall of 0.87 for class 1, while the [Neural Network model](https://www.kaggle.com/code/jbasurtod/predicting-obesity-with-neural-networks#Testing-the-Models) achieves a recall of 0.86 at its optimal threshold. For more details on their training, please refer to the linked Kaggle notebooks.
""")

# Load the models and scaler
@st.cache_resource
def load_models():
    # Load Random Forest model and unpack if necessary
    rf_model_tuple = joblib.load('models/rf_model.pkl')
    rf_model, _ = rf_model_tuple  # Unpack the model from the tuple
    # Load Neural Network model, scaler, and optimal threshold
    nn_model_tuple = joblib.load('models/NN_model.pkl')
    nn_model, scaler, optimal_threshold = nn_model_tuple
    return rf_model, nn_model, scaler, optimal_threshold

rf_model, nn_model, scaler, optimal_threshold = load_models()

# Dropdown menu for selecting the model
model_option = st.selectbox(
    "Choose the model to use for prediction:",
    options=["Random Forest", "Neural Network"]
)

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

# Button to send request and show result
if st.button("Predict"):
    # Prepare the data
    age_feature = np.array([Age]).reshape(1, -1)
    other_features = np.array([
        convert_yes_no(family_history_with_overweight_1),
        convert_yes_no(FAVC_1),
        convert_yes_no(CAEC_Sometimes),
        convert_yes_no(SCC_1)
    ]).reshape(1, -1)

    if model_option == "Random Forest":
        # Combine all features
        input_data = np.hstack([age_feature, other_features])
        # Predict with Random Forest model
        probabilities = rf_model.predict_proba(input_data)[0]
        prediction = np.argmax(probabilities)  # Get the index of the maximum probability
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]

    else:
        # Scale only the Age feature
        scaled_age = scaler.transform(age_feature)
        # Combine scaled Age with other features
        scaled_data = np.hstack([scaled_age, other_features])
        # Predict with Neural Network model
        probabilities = nn_model.predict(scaled_data)[0]
        # Apply the optimal threshold to determine the class
        prediction = (probabilities > optimal_threshold).astype(int)[0]
        # Use the probability of the positive class for displaying
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]

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
