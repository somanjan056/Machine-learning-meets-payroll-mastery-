import streamlit as st
import joblib
import pandas as pd

# Load the model (ensure best_model.pkl is in the same directory as app.py)
try:
    # Your best_model.pkl is a scikit-learn Pipeline, which includes preprocessing.
    model = joblib.load("best_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'best_model.pkl' is in the same directory and compatible with your scikit-learn version. Also check your Conda environment setup.")
    st.stop() # Stop execution if model can't be loaded

st.set_page_config(page_title="MY WEBPAGE", page_icon=":tada:", layout="wide")
st.subheader("Hi, I am Somanjan: wave.🖐")
st.title("A data analyst from India")

st.write("### Machine learning meets payroll mastery 💼")
st.write("Predict whether an employee earns >50K or <=50K based on input features.")

st.sidebar.header("Input Employee Details")

# --- INPUT WIDGETS FOR ALL ORIGINAL FEATURES ---
# Ensure widget names are user-friendly, but the resulting variables will map to model's feature names

# Numerical Features
age = st.sidebar.slider("Age", 17, 90, 30) # Adjusted range based on typical adult dataset
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=100000, min_value=1, max_value=1000000, help="This is a sampling weight; often not intuitive for direct user input but required by the model.")
capital_gain = st.sidebar.number_input("Capital Gain", value=0, min_value=0, max_value=100000, help="Income from investments/assets.")
capital_loss = st.sidebar.number_input("Capital Loss", value=0, min_value=0, max_value=100000, help="Loss from investments/assets.")
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40) # Adjusted range based on typical adult dataset
educational_num = st.sidebar.number_input("Educational Number", value=9, min_value=1, max_value=16, help="Numerical representation of education level (e.g., 9 for HS-grad, 13 for Bachelors).")


# Categorical Features (use exact values as they appear in the training data)
# These lists must be exhaustive for all categories seen during training.
# The 'education' and 'occupation' inputs from your previous code were good, just ensure column names match.

workclass_options = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"] # Added '?' if missing values were handled as such
workclass = st.sidebar.selectbox("Workclass", workclass_options)

education_options = [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
    "Assoc-acdm", "7th-8th", "Prof-school", "5th-6th", "10th", "1st-4th",
    "Doctorate", "Preschool", "12th", "Assoc-voc"
]
education = st.sidebar.selectbox("Education", education_options)

marital_status_options = [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
]
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)

occupation_options = [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces", "?" # Added '?' if missing values were handled as such
]
occupation = st.sidebar.selectbox("Occupation", occupation_options)

relationship_options = [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
]
relationship = st.sidebar.selectbox("Relationship", relationship_options)

race_options = [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
]
race = st.sidebar.selectbox("Race", race_options)

gender_options = ["Female", "Male"]
gender = st.sidebar.radio("Gender", gender_options)

native_country_options = [
    "United-States", "Cambodia", "England", "Puerto-Rico", "Germany",
    "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South",
    "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
    "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France",
    "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
    "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
    "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong",
    "Holand-Netherlands", "?" # Added '?' if missing values were handled as such
]
native_country = st.sidebar.selectbox("Native Country", native_country_options)


# --- CREATE DATAFRAME FOR PREDICTION ---
# IMPORTANT: The keys in this dictionary must EXACTLY match the original column names
# in your training data (X_train) before any preprocessing (like one-hot encoding).
# The Pipeline (model) will handle the transformations internally.
input_data_dict = {
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week], # Corrected: 'hours-per-week' is the actual column name
    'native-country': [native_country]
}

input_df = pd.DataFrame(input_data_dict)

# Display input data (optional, but good for debugging)
st.subheader("🔍 Input Data")
st.dataframe(input_df)

if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        # Assuming your model outputs 0 or 1, map to meaningful labels
        salary_class = ">50K" if prediction[0] == 1 else "<=50K"
        st.success(f"Predicted Salary Class: **{salary_class}**")
    except ValueError as ve:
        st.error(f"Prediction Error: {ve}")
        st.info("Please double-check that ALL input feature names and values EXACTLY match the model's expectations from its training data. Look for typos or missing features.")
        st.exception(ve) # Show full traceback for more detailed debugging
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        st.exception(e) # Show full traceback

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
st.markdown(
    """
    <style>
    .stApp {
        background-color: skyblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)