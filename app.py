import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import json
from passlib.hash import bcrypt
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
IMAGE_FILENAME = 'veterinary.jpg'
IMAGE_PATH = os.path.join(MODEL_ARTIFACTS_DIR, IMAGE_FILENAME)
USERS_FILE = os.path.join(MODEL_ARTIFACTS_DIR, 'users.json')

# --- HTML/CSS for Animated Bubble Background & Theme ---
# We'll use CSS variables for colors, which helps with theme adaptation.
# Streamlit injects its own theme, so we'll try to override/complement it.
BUBBLE_BACKGROUND_CSS = """
<style>
/* Define CSS Variables for adaptable colors */
:root {
    --primary-bg: #e0e6ed; /* Light mode body background */
    --secondary-bg: #ffffff; /* Light mode container/widget background */
    --text-color: #2c3e50; /* Dark text for light mode */
    --header-color: #1a2c3a; /* Darker headers */
    --accent-color: #4CAF50; /* Green accent for buttons/sliders */
    --bubble-color: rgba(200, 210, 220, 0.5); /* Semi-transparent light bubble */
    --bubble-border-color: rgba(180, 190, 200, 0.7); /* Slightly darker border for bubbles */
}

/* Dark mode overrides (Streamlit applies a [data-theme="dark"] attribute to html) */
html[data-theme="dark"] {
    --primary-bg: #1a1a2e; /* Darker background for dark mode */
    --secondary-bg: #2a2a4a; /* Darker container/widget background */
    --text-color: #e0e0e0; /* Lighter text for dark mode */
    --header-color: #f0f0f0; /* Lighter headers */
    --accent-color: #66BB6A; /* Brighter green accent for dark mode */
    --bubble-color: rgba(50, 60, 80, 0.5); /* Semi-transparent dark bubble */
    --bubble-border-color: rgba(70, 80, 100, 0.7); /* Slightly lighter border for dark bubbles */
}

body {
    background: var(--primary-bg); /* Use CSS variable */
    overflow: hidden; /* Hide scrollbars if bubbles go off screen */
    margin: 0;
    padding: 0;
}

/* The Streamlit main app container */
.stApp {
    background-color: transparent; /* Ensure transparent to show body background */
    color: var(--text-color); /* Apply text color variable */
}

/* Styling for titles and headers */
h1, h2, h3, h4, h5, h6 {
    color: var(--header-color); /* Apply header color variable */
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Subtle text shadow for pop */
}

/* General markdown text color */
.stMarkdown {
    color: var(--text-color);
}

/* Custom styling for Streamlit components */
.stSelectbox > div > div,
.stTextInput > div > div,
.stTextArea > div > div, /* Added textarea in case you add it later */
.stNumberInput > div > div {
    background-color: var(--secondary-bg); /* Use secondary background variable */
    border: 1px solid var(--bubble-border-color); /* Border based on bubble border for harmony */
    border-radius: 8px; /* Slightly more rounded */
    color: var(--text-color);
    padding: 8px 12px;
}

/* Style for sliders */
.stSlider .st-fx { /* Target the slider track */
    background-color: var(--accent-color); /* Accent color for track */
    border-radius: 5px;
}
.stSlider .st-fo { /* Target the slider thumb */
    background-color: var(--accent-color); /* Accent color for thumb */
    border: 2px solid var(--secondary-bg); /* White border for contrast */
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

/* Style for buttons */
.stButton > button {
    background-color: var(--accent-color);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 12px 25px;
    font-size: 17px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease; /* Smooth transition for hover effects */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
.stButton > button:hover {
    background-color: var(--header-color); /* Darker on hover */
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px); /* Slight lift on hover */
}

/* Style for containers used for visual grouping */
.stContainer {
    background-color: var(--secondary-bg); /* Use secondary background variable */
    padding: 30px; /* More padding */
    border-radius: 15px; /* More rounded */
    margin-bottom: 30px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15); /* More prominent shadow for depth */
    border: 1px solid rgba(0, 0, 0, 0.05); /* Very subtle border */
}

/* Specific styling for the prediction results container */
.prediction-results-container {
    background-color: rgba(144, 238, 144, 0.2); /* Very light green translucent */
    border: 1px solid var(--accent-color);
    color: var(--text-color);
}

/* --- Animated Bubbles Background --- */
.circles{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    z-index: -1; /* Send bubbles to the background */
}

.circles li{
    position: absolute;
    display: block;
    list-style: none;
    width: 20px;
    height: 20px;
    background: var(--bubble-color); /* Use bubble color variable */
    border: 1px solid var(--bubble-border-color); /* Use bubble border color variable */
    animation: animate 25s linear infinite;
    bottom: -150px; /* Start below the viewport */
    border-radius: 50%; /* Make them perfectly round */
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.2); /* Soft shadow for depth */
    opacity: 0; /* Start invisible */
}

.circles li:nth-child(1){
    left: 25%;
    width: 80px;
    height: 80px;
    animation-delay: 0s;
    animation-duration: 15s;
    opacity: 0.7;
}

.circles li:nth-child(2){
    left: 10%;
    width: 20px;
    height: 20px;
    animation-delay: 2s;
    animation-duration: 12s;
    opacity: 0.5;
}

.circles li:nth-child(3){
    left: 70%;
    width: 20px;
    height: 20px;
    animation-delay: 4s;
    animation-duration: 20s;
    opacity: 0.6;
}

.circles li:nth-child(4){
    left: 40%;
    width: 60px;
    height: 60px;
    animation-delay: 0s;
    animation-duration: 18s;
    opacity: 0.8;
}

.circles li:nth-child(5){
    left: 65%;
    width: 20px;
    height: 20px;
    animation-delay: 0s;
    animation-duration: 10s;
    opacity: 0.4;
}

.circles li:nth-child(6){
    left: 75%;
    width: 110px;
    height: 110px;
    animation-delay: 3s;
    animation-duration: 22s;
    opacity: 0.9;
}

.circles li:nth-child(7){
    left: 35%;
    width: 150px;
    height: 150px;
    animation-delay: 7s;
    animation-duration: 25s;
    opacity: 0.7;
}

.circles li:nth-child(8){
    left: 50%;
    width: 25px;
    height: 25px;
    animation-delay: 15s;
    animation-duration: 45s;
    opacity: 0.5;
}

.circles li:nth-child(9){
    left: 20%;
    width: 15px;
    height: 15px;
    animation-delay: 2s;
    animation-duration: 35s;
    opacity: 0.6;
}

.circles li:nth-child(10){
    left: 85%;
    width: 150px;
    height: 150px;
    animation-delay: 0s;
    animation-duration: 11s;
    opacity: 0.8;
}

/* The actual animation */
@keyframes animate {
    0%{
        transform: translateY(0) rotate(0deg);
        opacity: 0.8;
        border-radius: 0;
    }
    100%{
        transform: translateY(-1000px) rotate(720deg);
        opacity: 0;
        border-radius: 50%;
    }
}
</style>
"""

# --- Initial Streamlit Page Setup (MUST be at the very top, called only once) ---
st.set_page_config(page_title="Brucellosis Prediction App", layout="wide")

# Inject the custom CSS for the animated bubble background and styling
st.markdown(BUBBLE_BACKGROUND_CSS, unsafe_allow_html=True)

# Add the HTML structure for the bubbles
st.markdown("""
<div class="circles">
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
</div>
""", unsafe_allow_html=True)

# --- LOAD USER CREDENTIALS ---
users = {}
try:
    if not os.path.exists(USERS_FILE):
        st.sidebar.error(f"❌ User credentials file not found at '{USERS_FILE}'. Please create it with hashed passwords.")
        st.stop()
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    st.sidebar.success("🔒 User credentials loaded successfully!")
except json.JSONDecodeError:
    st.sidebar.error(f"❌ Error decoding '{USERS_FILE}'. Ensure it's valid JSON (e.g., check for missing commas, brackets).")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Error loading user credentials: {e}")
    st.stop()

# --- INITIALIZE SESSION STATE FOR LOGIN ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- LOAD MODEL ARTIFACTS ---
try:
    required_model_files = ['best_model.pkl', 'le_dict.pkl', 'le_target.pkl', 'scaler.pkl', 'feature_names.pkl']
    for filename in required_model_files:
        file_path = os.path.join(MODEL_ARTIFACTS_DIR, filename)
        if not os.path.exists(file_path):
            st.sidebar.error(f"❌ Required model file not found: '{file_path}'. Please ensure all .pkl files are in the '{MODEL_ARTIFACTS_DIR}' directory.")
            st.stop()

    with open(os.path.join(MODEL_ARTIFACTS_DIR, 'best_model.pkl'), 'rb') as f:
        best_model = pickle.load(f)
    with open(os.path.join(MODEL_ARTIFACTS_DIR, 'le_dict.pkl'), 'rb') as f:
        le_dict = pickle.load(f)
    with open(os.path.join(MODEL_ARTIFACTS_DIR, 'le_target.pkl'), 'rb') as f:
        le_target = pickle.load(f)
    with open(os.path.join(MODEL_ARTIFACTS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_ARTIFACTS_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)

    st.sidebar.success("✅ All model components loaded successfully!")

except Exception as e:
    st.sidebar.error(f"❌ Error loading model components: {e}")
    st.stop()

# --- AUTHENTICATION FUNCTION ---
def login_page():
    st.sidebar.subheader("Login")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if email in users:
            try:
                if bcrypt.verify(password, users[email]):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = email
                    st.sidebar.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid email or password.")
                    st.session_state['logged_in'] = False
            except ValueError as e:
                st.sidebar.error("Invalid credentials format or internal error. Please contact support.")
                st.exception(e)
                st.session_state['logged_in'] = False
        else:
            st.sidebar.error("Invalid email or password.")
            st.session_state['logged_in'] = False

    st.sidebar.markdown("---")
    st.sidebar.info("Please enter your registered email and password to access the app.")

# --- MAIN APP LOGIC (Conditional rendering based on login status) ---
if not st.session_state['logged_in']:
    st.title("Welcome to Brucellosis Prediction App")
    if os.path.exists(IMAGE_PATH):
        try:
            st.image(IMAGE_PATH, caption="Caring for Animal Health", use_column_width=True)
        except Exception as e:
            st.error(f"⚠️ Error displaying veterinary image: {e}")
    else:
        st.warning(f"⚠️ Veterinary image not found at '{IMAGE_PATH}'. Ensure '{IMAGE_FILENAME}' is in '{MODEL_ARTIFACTS_DIR}'.")

    login_page()
else:
    st.title("🐂 Brucellosis Prediction Model")
    st.markdown(f"Welcome, **{st.session_state['username']}**! Enter the animal's details to predict its Brucellosis status.")

    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False, username=None))
    st.sidebar.markdown("---")

    unique_breeds = sorted(list(le_dict.get('Breed species', LabelEncoder()).classes_))
    unique_sex = sorted(list(le_dict.get('Sex', LabelEncoder()).classes_))
    unique_abortion_history = sorted(list(le_dict.get('Abortion History (Yes No)', LabelEncoder()).classes_))
    unique_infertility = sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)', LabelEncoder()).classes_))
    unique_vaccination_status = sorted(list(le_dict.get('Brucella vaccination status (Yes No)', LabelEncoder()).classes_))
    unique_sample_type = sorted(list(le_dict.get('Sample Type(Serum Milk)', LabelEncoder()).classes_))
    unique_test_type = sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)', LabelEncoder()).classes_))
    unique_retained_placenta = sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)', LabelEncoder()).classes_))
    unique_disposal = sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)', LabelEncoder()).classes_))


    def predict_single_case(input_dict, model, le_dict, le_target, scaler, feature_names):
        input_df = pd.DataFrame([input_dict])
        input_df.columns = input_df.columns.str.strip()

        if 'Breed species' in input_df.columns:
            input_df['Breed species'] = input_df['Breed species'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

        for col in input_df.columns:
            if col in le_dict and input_df.dtypes.get(col) == 'object':
                try:
                    input_df.loc[:, col] = le_dict.get(col).transform(input_df.loc[:, col])
                except ValueError as e:
                    st.error(f"❌ Error encoding column '{col}': The input value '{input_dict.get(col)}' is not a known category. Known categories: {list(le_dict.get(col).classes_)}")
                    return None

        for col in feature_names:
            if col not in input_df.columns:
                input_df.loc[:, col] = 0

        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        model_requires_scaling = isinstance(model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))
        input_df_scaled = scaler.transform(input_df) if model_requires_scaling else input_df.values

        pred_class = model.predict(input_df_scaled)[0]
        pred_prob = model.predict_proba(input_df_scaled)[0]

        predicted_result = le_target.inverse_transform([pred_class])[0]
        confidence = pred_prob.max()

        return {
            'predicted_class': predicted_result,
            'confidence': confidence,
            'probabilities': dict(zip(le_target.classes_, pred_prob))
        }

    st.sidebar.header("Input Features")

    with st.container():
        st.subheader("Input Animal Details")
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age (Years)", 0, 20, 5)
            breed_species = st.selectbox("Breed/Species", options=unique_breeds)
            sex = st.selectbox("Sex", options=unique_sex)
            calvings = st.slider("Calvings", 0, 15, 1)
            abortion_history = st.selectbox("Abortion History (Yes/No)", options=unique_abortion_history)

        with col2:
            infertility_rb = st.selectbox("Infertility/Repeat Breeder (Yes/No)", options=unique_infertility)
            vaccination_status = st.selectbox("Brucella Vaccination Status (Yes/No)", options=unique_vaccination_status)
            sample_type = st.selectbox("Sample Type (Serum/Milk)", options=unique_sample_type)
            test_type = st.selectbox("Test Type (RBPT/ELISA/MRT)", options=unique_test_type)
            retained_placenta = st.selectbox("Retained Placenta/Stillbirth", options=unique_retained_placenta)
            proper_disposal = st.selectbox("Proper Disposal of Aborted Fetuses (Yes No)", options=unique_disposal)

        input_data = {
            'Age': age,
            'Breed species': breed_species,
            'Sex': sex,
            'Calvings': calvings,
            'Abortion History (Yes No)': abortion_history,
            'Infertility Repeat breeder(Yes No)': infertility_rb,
            'Brucella vaccination status (Yes No)': vaccination_status,
            'Sample Type(Serum Milk)': sample_type,
            'Test Type (RBPT ELISA MRT)': test_type,
            'Retained Placenta Stillbirth(Yes No No Data)': retained_placenta,
            'Proper Disposal of Aborted Fetuses (Yes No)': proper_disposal
        }

    st.subheader("Provided Input:")
    # This st.json output often looks better in its default dark background,
    # let's try not to apply a background-color to it via generic containers
    st.json(input_data)

    if st.button("Predict Brucellosis Status"):
        with st.container(): # Use container here for visual grouping of results
            st.subheader("Prediction Results:")
            with st.spinner('Making prediction...'):
                prediction_output = predict_single_case(
                    input_data, best_model, le_dict, le_target, scaler, feature_names
                )

                if prediction_output:
                    st.success(f"**Predicted Result:** {prediction_output['predicted_class']}")
                    st.info(f"**Confidence:** {prediction_output['confidence']:.2%}")

                    st.write("---")
                    st.subheader("Class-wise Probabilities:")
                    prob_df = pd.DataFrame.from_dict(prediction_output['probabilities'], orient='index', columns=['Probability'])
                    prob_df = prob_df.sort_values(by='Probability', ascending=False)
                    st.dataframe(prob_df.style.format("{:.2%}"))

                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=prob_df.index, y=prob_df['Probability'], palette='viridis', ax=ax)
                    ax.set_title("Predicted Class Probabilities")
                    ax.set_ylabel("Probability")
                    ax.set_xlabel("Brucellosis Status")
                    st.pyplot(fig)
                else:
                    st.error("Failed to make a prediction. Please check the input values and error messages above.")

    st.markdown("---")
    st.markdown("Developed with ❤️ for Veterinary Health")
