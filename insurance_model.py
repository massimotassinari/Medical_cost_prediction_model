import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error
import streamlit as st
from joblib import dump,load

# Function to calculate BMI
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100  # Convert height to meters
    return weight_kg / (height_m ** 2)

def center_title(title):
    col1, col2,col3 = st.columns([1,7,1])
    with col1:
        pass
    with col2:
        st.title(title,anchor='center')
    with col3:
        pass

def center_header(header):
    col1, col2,col3 = st.columns([1, 30,1])
    with col1:
        pass
    with col2:
        st.title(header,anchor='center')
    with col3:
        pass

def model(input_data):
    # Load the scaler and model
    scaler = load('./saved_model/scaler.joblib')
    loaded_model = load('./saved_model/insurance_model.joblib')

    # Apply the same transformation to input data
    input_data = scaler.transform(input_data)

    # Make predictions
    prediction = loaded_model.predict(input_data)

    return prediction

def main():
    # Define the Streamlit app
    center_title("Insurance Prediction App")
    # Add a picture at the top
    st.image('./media/insurance.jpg')

    # Create two larger columns with equal width
    col1, col2 = st.columns([3, 2])  # Adjust column proportions as needed

    # Column for inputs
    with st.container():
        with col1:
            center_header("Enter Your Details")

            # Collect user input for model parameters
            age = st.slider("Age", min_value=18, max_value=100, value=19)
            bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=27.9, step=0.1)
            with st.expander('Calculate your BMI'):
                # User inputs
  
                weight = st.number_input("Enter your weight (kg):", min_value=1, step=1, format="%d")
                height = st.number_input("Enter your height (cm):", min_value=1, step=1, format="%d")

                # Calculate BMI when the button is pressed
                if st.button("Calculate BMI"):
                    if weight > 0 and height > 0:
                        bmi = calculate_bmi(weight, height)
                        st.success(f"Your BMI is: {bmi:.1f} kg/m²")
                        
                        # Display BMI category
                        if bmi < 18.5:
                            st.info("Category: Underweight")
                        elif 18.5 <= bmi < 25:
                            st.info("Category: Normal weight")
                        elif 25 <= bmi < 30:
                            st.warning("Category: Overweight")
                        else:
                            st.error("Category: Obesity")
                    else:
                        st.error("Please enter valid weight and height.")
            children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

            # Radio button for smoker status
            smoker = st.radio("Are you a smoker?", ["No", "Yes"])
            smoker_no = 1 if smoker == "No" else 0
            smoker_yes = 1 if smoker == "Yes" else 0

            # Radio button for gender
            sex = st.radio("Sex", ["Female", "Male"])
            sex_female = 1 if sex == "Female" else 0
            sex_male = 1 if sex == "Male" else 0

            # Radio button for region
            region = st.radio("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
            region_northeast = 1 if region == "Northeast" else 0
            region_northwest = 1 if region == "Northwest" else 0
            region_southeast = 1 if region == "Southeast" else 0
            region_southwest = 1 if region == "Southwest" else 0

            # Assemble input data in the specified format
            input_data = [[
                age, bmi, children, smoker_no, smoker_yes, sex_female, sex_male,
                region_northeast, region_northwest, region_southeast, region_southwest
            ]]

        # Column for output
        with col2:
            center_header("Prediction")

            col1, col2, col3 = st.columns([1, 9, 1])  # Adjust column proportions as needed
            with col1:
                pass
            with col2:
                # Make prediction when user clicks the button
                if st.button("Predict Insurance Cost"):
                    prediction = model(input_data)  # Replace with your actual model function

        
                    # Check the prediction value and show the appropriate image and animation
                    if prediction[0] <= 0:
                        st.markdown(
                        f"<h1 style='text-align: center; color: green; font-size: 40px;'>"
                        f"Base Line Price</h1>",
                        unsafe_allow_html=True
                    )
                        st.image('./media/good.png')
                        #st.success("This insurance lower than the mean!")  # Display an error message
                        st.balloons()  # Show balloons if the prediction is good

                    elif prediction[0] < 13270.422265141257:
                        st.markdown(
                        f"<h1 style='text-align: center; color: green; font-size: 40px;'>"
                        f"${prediction[0]:,.2f}</h1>",
                        unsafe_allow_html=True
                    )
                        st.image('./media/good.png')
                        st.success("This insurance lower than the mean!")  # Display an error message
                        st.balloons()  # Show balloons if the prediction is good
                    else:
                        st.markdown(
                        f"<h1 style='text-align: center; color: red; font-size: 40px;'>"
                        f"${prediction[0]:,.2f}</h1>",
                        unsafe_allow_html=True
                        )
                        st.image('./media/bad.png' )
                        st.error("This insurance prediction is higher than the mean!")  # Display an error message

            with col3:
                pass

        with st.expander('Lets see an example'):
	        st.write("""
		insert a multi element container that can be expanded or colapsed 
		by the user """)
            

st.markdown('---')

# Run the app
if __name__ == "__main__":
    main()
