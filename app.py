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
from sklearn.neighbors import KNeighborsClassifier
import json
from passlib.hash import pbkdf2_sha256 

warnings.filterwarnings('ignore')

# --- TRANSLATIONS DICTIONARY ---
translations = {
    "English": {
        "welcome": "Welcome to Brucellosis Prediction App",
        "title": "🐂 Brucellosis Prediction Model",
        "user_greet": "Welcome, **{}**! Enter the animal's details to predict its Brucellosis status.",
        "input_header": "Input Features",
        "age": "Age (Years)",
        "breed": "Breed/Species",
        "sex": "Sex",
        "calvings": "Calvings",
        "abortion": "Abortion History (Yes/No)",
        "infertility": "Infertility/Repeat Breeder (Yes/No)",
        "vaccination": "Brucella Vaccination Status (Yes/No)",
        "sample": "Sample Type (Serum/Milk)",
        "test": "Test Type (RBPT/ELISA/MRT)",
        "retained": "Retained Placenta/Stillbirth",
        "disposal": "Proper Disposal of Aborted Fetuses (Yes No)",
        "predict_btn": "Predict Brucellosis Status",
        "provided_input": "Provided Input:",
        "results_header": "Prediction Results:",
        "pred_res": "**Predicted Result:**",
        "conf": "**Confidence:**",
        "prob_header": "Class-wise Probabilities:",
        "chart_title": "Predicted Class Probabilities",
        "logout": "Logout",
        "login_sub": "Login",
        "lang_label": "Choose Language / भाषा चुनें"
    },
    "Hindi": {
        "welcome": "ब्रुसेलोसिस भविष्यवाणी ऐप में आपका स्वागत है",
        "title": "🐂 ब्रुसेलोसिस भविष्यवाणी मॉडल",
        "user_greet": "आपका स्वागत है, **{}**! भविष्यवाणी करने के लिए पशु का विवरण दर्ज करें।",
        "input_header": "इनपुट विशेषताएं",
        "age": "आयु (वर्ष)",
        "breed": "नस्ल/प्रजाति",
        "sex": "लिंग",
        "calvings": "बछड़े की संख्या (Calvings)",
        "abortion": "गर्भपात का इतिहास (हाँ/नहीं)",
        "infertility": "बांझपन (हाँ/नहीं)",
        "vaccination": "ब्रुसेला टीकाकरण की स्थिति (हाँ/नहीं)",
        "sample": "नमूना प्रकार (सीरम/दूध)",
        "test": "परीक्षण प्रकार (RBPT/ELISA/MRT)",
        "retained": "जेर रुकना/मृत प्रसव (Retained Placenta)",
        "disposal": "गर्भपात भ्रूण का उचित निपटान",
        "predict_btn": "स्थिति की भविष्यवाणी करें",
        "provided_input": "प्रदान किया गया इनपुट:",
        "results_header": "भविष्यवाणी के परिणाम:",
        "pred_res": "**अनुमानित परिणाम:**",
        "conf": "**भरोसा (Confidence):**",
        "prob_header": "वर्ग-वार संभावनाएं:",
        "chart_title": "अनुमानित वर्ग संभावनाएं",
        "logout": "लॉगआउट",
        "login_sub": "लॉगिन",
        "lang_label": "भाषा चुनें"
    }
}

# --- CONFIGURATION ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'

# --- INITIALIZE SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- LOAD USER CREDENTIALS ---
users = {}
try:
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
except Exception:
    st.sidebar.error("❌ User credentials file error.")

# --- LOAD MODEL ARTIFACTS ---
try:
    with open(MODEL_ARTIFACTS_DIR + 'best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'le_dict.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'le_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- LANGUAGE SELECTION ---
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
t = translations[selected_lang]

# --- AUTHENTICATION FUNCTION ---
def login_page():
    st.sidebar.subheader(t["login_sub"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if email in users and pbkdf2_sha256.verify(password, users[email]):
            st.session_state['logged_in'] = True
            st.session_state['username'] = email
            st.rerun()
        else:
            st.sidebar.error("Invalid email or password.")

# --- MAIN APP LOGIC ---
st.set_page_config(page_title="Brucellosis Prediction App", layout="wide")

if not st.session_state['logged_in']:
    st.title(t["welcome"])
    login_page()
else:
    st.title(t["title"])
    st.markdown(t["user_greet"].format(st.session_state['username']))

    st.sidebar.button(t["logout"], on_click=lambda: st.session_state.update(logged_in=False, username=None))
    
    # Get unique categories
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
                except ValueError:
                    return None

        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        model_requires_scaling = isinstance(model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))
        input_data_processed = scaler.transform(input_df) if model_requires_scaling else input_df.values

        pred_class = model.predict(input_data_processed)[0]
        pred_prob = model.predict_proba(input_data_processed)[0]
        return {
            'predicted_class': le_target.inverse_transform([pred_class])[0],
            'confidence': pred_prob.max(),
            'probabilities': dict(zip(le_target.classes_, pred_prob))
        }

    st.sidebar.header(t["input_header"])

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider(t["age"], 0, 20, 5)
        breed_species = st.selectbox(t["breed"], options=unique_breeds)
        sex = st.selectbox(t["sex"], options=unique_sex)
        calvings = st.slider(t["calvings"], 0, 15, 1)
        abortion_history = st.selectbox(t["abortion"], options=unique_abortion_history)

    with col2:
        infertility_rb = st.selectbox(t["infertility"], options=unique_infertility)
        vaccination_status = st.selectbox(t["vaccination"], options=unique_vaccination_status)
        sample_type = st.selectbox(t["sample"], options=unique_sample_type)
        test_type = st.selectbox(t["test"], options=unique_test_type)
        retained_placenta = st.selectbox(t["retained"], options=unique_retained_placenta)
        proper_disposal = st.selectbox(t["disposal"], options=unique_disposal)

    input_data = {
        'Age': age, 'Breed species': breed_species, 'Sex': sex, 'Calvings': calvings,
        'Abortion History (Yes No)': abortion_history, 'Infertility Repeat breeder(Yes No)': infertility_rb,
        'Brucella vaccination status (Yes No)': vaccination_status, 'Sample Type(Serum Milk)': sample_type,
        'Test Type (RBPT ELISA MRT)': test_type, 'Retained Placenta Stillbirth(Yes No No Data)': retained_placenta,
        'Proper Disposal of Aborted Fetuses (Yes No)': proper_disposal
    }

    st.subheader(t["provided_input"])
    st.json(input_data)

    if st.button(t["predict_btn"]):
        st.subheader(t["results_header"])
        with st.spinner('Predicting...'):
            output = predict_single_case(input_data, best_model, le_dict, le_target, scaler, feature_names)
            if output:
                # Result logic
                res_val = output['predicted_class']
                # Translate "Positive/Negative" result if Hindi
                if selected_lang == "Hindi":
                    res_val = "पॉजिटिव (Positive)" if "Positive" in res_val else "नेगेटिव (Negative)"
                
                st.success(f"{t['pred_res']} {res_val}")
                st.info(f"{t['conf']} {output['confidence']:.2%}")

                st.write("---")
                st.subheader(t["prob_header"])
                prob_df = pd.DataFrame.from_dict(output['probabilities'], orient='index', columns=['Probability'])
                st.dataframe(prob_df.style.format("{:.2%}"))

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=prob_df.index, y=prob_df['Probability'], palette='viridis', ax=ax)
                ax.set_title(t["chart_title"])
                st.pyplot(fig)

    st.markdown("---")
    st.markdown("Developed with ❤️ for Veterinary Health")
