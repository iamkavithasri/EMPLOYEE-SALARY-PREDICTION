import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

# Mapping dictionaries (update based on your actual encodings)
marital_status_map = {
    "Married": 1,
    "Single": 2,
    "Divorced": 3,
    "Separated": 4,
    "Widowed": 5
}

# Mapping from education labels to numbers (based on UCI dataset)
education_map = {

    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}

workclass_map = {
    "Private": 1,
    "Self-emp-not-inc": 2,
    "Self-emp-inc": 3,
    "Federal-gov": 4,
    "Local-gov": 5,
    "State-gov": 6,
   
   
}


occupation_map = {
    "Tech-support": 1,
    "Craft-repair": 2,
    "Other-service": 3,
    "Sales": 4,
    "Exec-managerial": 5,
    "Prof-specialty": 6,
    "Handlers-cleaners": 7,
    "Machine-op-inspct": 8,
    "Adm-clerical": 9,
    "Farming-fishing": 10,
    "Transport-moving": 11,
    "Priv-house-serv": 12,
    "Protective-serv": 13,
    "Armed-Forces": 14
}

relationship_map = {
    "Husband": 1,
    "Wife": 2,
    "Not-in-family": 3,
    "Own-child": 4,
    "Unmarried": 5,
    "Other-relative": 6
}

race_map = {
    "White": 1,
    "Black": 2,
    "Asian-Pac-Islander": 3,
    "Amer-Indian-Eskimo": 4,
    "Other": 5
}

gender_map = {
    "Male": 1,
    "Female": 0
}

native_country_map = {
    "United-States": 39,
    "India": 25,
    "Philippines": 31,
    "Germany": 14,
    "Mexico": 29,
    "Other": 1
}

# Streamlit UI setup
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Collect user input
age = st.sidebar.slider("Age", 18, 65, 30)

workclass_label = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
workclass = workclass_map[workclass_label]

education_label = st.sidebar.selectbox("Education Level", list(education_map.keys()))
educational_num = education_map[education_label]

marital_status_label = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
marital_status = marital_status_map[marital_status_label]

occupation_label = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
occupation = occupation_map[occupation_label]

relationship_label = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
relationship = relationship_map[relationship_label]

race_label = st.sidebar.selectbox("Race", list(race_map.keys()))
race = race_map[race_label]

gender_label = st.sidebar.selectbox("Gender", list(gender_map.keys()))
gender = gender_map[gender_label]

capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)

hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)

native_country_label = st.sidebar.selectbox("Native Country", list(native_country_map.keys()))
native_country = native_country_map[native_country_label]

# Build input dataframe
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')


