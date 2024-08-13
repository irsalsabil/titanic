import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('titanic_model.pkl')

# Set the title and favicon for the Browser's tab bar.
st.set_page_config(
    page_title='Titanic Survivor Prediction',
    page_icon=':ship:',  # This is an emoji shortcode. Could be a URL too.
)

# Set the title that appears at the top of the page.
st.markdown('''
# :ship: Titanic Survivor Prediction

This application allows you to predict whether a passenger would have survived the Titanic disaster based on various features. Using a trained machine learning model, you can input passenger details such as class, age, sex, number of siblings/spouses, number of parents/children, fare paid, and port of embarkation to get a survival prediction along with the probability of survival.
''')

def user_input_features():
    st.sidebar.header('Passenger Information')
    
    # Collect user inputs with clear explanations
    pclass = st.sidebar.selectbox(
        'Select the class of the passenger:',
        ['1st Class', '2nd Class', '3rd Class'],
        help="Choose the class of the passenger. 1st class is the most expensive, 2nd class is mid-range, and 3rd class is the least expensive."
    )
    
    sex = st.sidebar.selectbox(
        'Select the gender of the passenger:',
        ['Male', 'Female'],
        help="Choose the gender of the passenger."
    )
    
    age = st.sidebar.slider(
        'Select the age of the passenger:',
        0, 100, 30,
        help="Slide to specify the age of the passenger. Ages range from 0 to 100."
    )
    
    sibsp = st.sidebar.slider(
        'Select the number of siblings or spouses aboard:',
        0, 10, 1,
        help="Slide to indicate the number of siblings or spouses the passenger has aboard. This includes brothers, sisters, and spouses."
    )
    
    parch = st.sidebar.slider(
        'Select the number of parents or children aboard:',
        0, 10, 0,
        help="Slide to specify the number of parents or children the passenger has aboard. This includes mothers, fathers, sons, and daughters."
    )
    
    fare = st.sidebar.number_input(
        'Enter the amount of fare paid by the passenger:',
        0.0, 500.0, 30.0,
        help="Enter the amount of fare paid by the passenger. This value ranges from 0.0 to 500.0."
    )
    
    embarked = st.sidebar.selectbox(
        'Select the port where the passenger boarded the Titanic:',
        ['Cherbourg', 'Queenstown', 'Southampton'],
        help="Choose the port of embarkation."
    )

    # Convert user input to DataFrame
    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user inputs
st.subheader("Passenger Information", divider='gray')
st.write("_Input the passenger detail in the sidebar_")
st.write(input_df)

# Preprocess input for model prediction
def preprocess_input(input_df):
    # Example preprocessing steps - modify according to your model's requirements
    input_df['Pclass'] = input_df['Pclass'].map({'1st Class': 1, '2nd Class': 2, '3rd Class': 3})
    input_df['Sex'] = input_df['Sex'].map({'Male': 0, 'Female': 1})
    input_df['Embarked'] = input_df['Embarked'].map({'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2})
    return input_df

processed_input = preprocess_input(input_df)

# Make prediction
prediction = model.predict(processed_input)
prediction_proba = model.predict_proba(processed_input)

# Display prediction
st.subheader("Prediction", divider='gray')
survived = "You're probably going to survive" if prediction[0] == 1 else "You're probably not going to survive"
st.write(f"##### {survived}")

st.subheader("Survival Probability", divider='gray')
st.write(f"##### {prediction_proba[0][1]:.2f}")

# Add credits in the sidebar
st.sidebar.markdown("##")
st.sidebar.markdown("### Crafted by:")
st.sidebar.markdown("**Irsa Salsabillah**")