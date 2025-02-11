# Medical Cost Prediction Model

## Overview
The **Medical Cost Prediction Model** is a machine learning application designed to predict health insurance costs based on various user-provided factors such as age, BMI, smoking status, and region. The prediction model is deployed as an interactive Streamlit web app.

### Link to Streamlit App
[Medical Cost Prediction App](https://medicalcostpredictionmodel-h5xdteytkwcpx7ondztkat.streamlit.app/)

---

## Features

### 1. User Input
Users can provide the following information:
- Age
- Body Mass Index (BMI) or calculate BMI based on weight and height
- Number of children
- Smoking status (Yes/No)
- Gender (Male/Female)
- Geographic region (Northeast, Northwest, Southeast, Southwest)

### 2. Prediction
The app displays a prediction of the insurance cost. Depending on the prediction, users will see visual feedback:
- Green and success message if the prediction is below the mean insurance cost.
- Red and error message if the prediction exceeds the mean.

### 3. Visual Interface
The app uses Streamlit's layout features to provide an intuitive and interactive interface.

---

## Technical Details

### Model Training
- The training script (`model_training.py`) uses the ElasticNet regression model with a hyperparameter grid search to optimize predictions.
- Input data undergoes preprocessing, including one-hot encoding for categorical variables (sex, smoker, and region).
- The dataset is split into training and test sets, and feature scaling is applied using `StandardScaler`.
- The trained model and scaler are saved as `.joblib` files for use in the prediction app.

### Model Inference
- The app (`insurance_model.py`) loads the pre-trained model and scaler.
- User input data is transformed to match the training data structure and scaled before making predictions.
- Streamlit components are used to display input fields, predictions, and feedback.

### Data Source
The dataset (`insurance.csv`) contains information on:
- Age, sex, BMI, number of children, smoking status, region, and insurance charges.

---

## Dependencies
All required libraries are listed in `requirements.txt`:
- Streamlit
- NumPy
- pandas
- scikit-learn
- seaborn
- matplotlib
- plotly
- joblib

To install the dependencies, run:
```sh
pip install -r requirements.txt
```

---

## Directory Structure
```plaintext
project/
    |-- insurance.csv           # Dataset
    |-- insurance_model.py      # Streamlit app script
    |-- model_training.py       # Model training script
    |-- README.md               # Project documentation
    |-- requirements.txt        # Dependency list
```

---

## How to Run the App Locally
1. Clone the project repository.
2. Install the dependencies.
3. Run the Streamlit app:
   ```sh
   streamlit run insurance_model.py
   ```
4. Access the app locally via the URL provided by Streamlit.

---

## Future Enhancements
- Improve model accuracy with additional features or alternative models.
- Add support for other healthcare cost predictors.
- Enable advanced analytics for deeper insights into cost factors.
