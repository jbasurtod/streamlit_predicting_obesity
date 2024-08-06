# Streamlit Obesity Prediction App

## Overview

This Streamlit app is designed to predict the likelihood of obesity based on various health and lifestyle criteria. A Random Forest classifier and a Neural Network were trained using the "Estimation of Obesity Levels Based On Eating Habits and Physical Condition" dataset from the UCI Machine Learning Repository.

## Explore Further

- **Kaggle Notebook**: Dive deeper into the analysis and code by visiting the Kaggle notebooks for the [Random Forest](https://www.kaggle.com/code/jbasurtod/predicting-obesity-with-random-forests) and [Neural Network](https://www.kaggle.com/code/jbasurtod/predicting-obesity-with-neural-networks) models training. 
- **Live Model**: Experience the model in action on the Streamlit app [here](https://obesitypred.streamlit.app/).

## App Features

- **Age**: User's age
- **Family History with Overweight**: Whether the user has a family history of overweight (Yes/No)
- **Frequent Consumption of Fruits and Vegetables**: Whether the user frequently consumes fruits and vegetables (Yes/No)
- **Sometimes Consumes High Caloric Food**: Whether the user sometimes consumes high caloric food (Yes/No)
- **Daily Caloric Monitoring**: Whether the user monitors their daily caloric intake (Yes/No)

## Model Performance

- **Random Forest Recall**: 0.87 in identifying obesity
- **Neural Network Recall**: 0.94 in identifying obesity

## Dataset

This app uses the "Estimation of Obesity Levels Based On Eating Habits and Physical Condition" dataset, which can be found [here](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jbasurtod/streamlit_predicting_obesity.git
   cd streamlit_predicting_obesity

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit app**
   ```bash
   streamlit run dashboard.py

## How to Use

- Open the app in your browser by visiting the URL provided by Streamlit after running the above command (typically http://localhost:8501).
- Enter your age and respond to the questions about your lifestyle.
- Click the "Predict" button to see the prediction and the probability of being obese.

## Deployment
This app can be deployed on Streamlit Cloud. Make sure to connect your GitHub repository to Streamlit Cloud for automatic deployments.

## Author and License

Created by Juan C Basurto. This project is licensed under the MIT License.
