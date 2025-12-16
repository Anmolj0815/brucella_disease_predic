import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from passlib.hash import pbkdf2_sha256 

warnings.filterwarnings('ignore')

# --- GEMINI CONFIGURATION ---
ai_enabled = False
if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # FIX: Using 'gemini-1.5-flash-latest' often resolves versioning 404s
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        ai_enabled = True
    except Exception as e:
        st.sidebar.error(f"AI Config Error: {e}")
else:
    st.sidebar.warning("⚠️ API Key missing in Streamlit Secrets.")

# --- TRANSLATIONS ---
translations = {
    "English": {
        "welcome": "Welcome to Brucellosis Prediction App",
        "title": "🐂 Brucellosis Prediction Model",
        "user_greet": "Welcome, **{}**!",
        "predict_btn": "Predict Brucellosis Status",
        "results_header": "Prediction Results:",
        "ai_advice_header": "🤖 AI Veterinary Consultation",
        "ai_loading": "Generating suggestions...",
        "system_prompt": "As a vet expert, analyze: {}. Result: {}. Confidence: {}%. Provide 3 steps for the farmer in English."
    },
    "Hindi": {
        "welcome": "ब्रुसेलोसिस भविष्यवाणी ऐप में आपका स्वागत है",
        "title": "🐂 ब्रुसेलोसिस भविष्यवाणी मॉडल",
        "user_greet": "आपका स्वागत है, **{}**!",
        "predict_btn": "भविष्यवाणी करें",
        "results_header": "परिणाम:",
        "ai_advice_header": "🤖 AI पशु चिकित्सक सलाह",
        "ai_loading": "सुझाव तैयार किए जा रहे हैं...",
        "system_prompt": "पशु चिकित्सक के रूप में, विश्लेषण करें: {}. परिणाम: {}. भरोसा: {}%. किसान के लिए हिंदी में 3 सुझाव दें।"
    }
}

# --- ARTIFACT LOADING ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'

@st.cache_resource
def load_artifacts():
    with open(MODEL_ARTIFACTS_DIR + 'best_model.pkl', 'rb') as f: m = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'le_dict.pkl', 'rb') as f: ld = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'le_target.pkl', 'rb') as f: lt = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'scaler.pkl', 'rb') as f: s = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'feature_names.pkl', 'rb') as f: fn = pickle.load(f)
    return m, ld, lt, s, fn

best_model, le_dict, le_target, scaler, feature_names = load_artifacts()

# --- APP LOGIC ---
st.set_page_config(page_title="Brucella AI", layout="wide")
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
t = translations[selected_lang]

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title(t["welcome"])
    email = st.sidebar.text_input("Email")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        with open(USERS_FILE, 'r') as f: users = json.load(f)
        if email in users and pbkdf2_sha256.verify(pwd, users[email]):
            st.session_state.update(logged_in=True, username=email)
            st.rerun()
else:
    st.title(t["title"])
    # Features UI
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 0, 20, 5)
        breed = st.selectbox("Breed", options=list(le_dict.get('Breed species').classes_))
        sex = st.selectbox("Sex", options=list(le_dict.get('Sex').classes_))
        calvings = st.slider("Calvings", 0, 15, 1)
        abortion = st.selectbox("Abortion History", options=list(le_dict.get('Abortion History (Yes No)').classes_))
    with col2:
        infertility = st.selectbox("Infertility", options=list(le_dict.get('Infertility Repeat breeder(Yes No)').classes_))
        vaccine = st.selectbox("Vaccination", options=list(le_dict.get('Brucella vaccination status (Yes No)').classes_))
        sample = st.selectbox("Sample Type", options=list(le_dict.get('Sample Type(Serum Milk)').classes_))
        test_type = st.selectbox("Test Type", options=list(le_dict.get('Test Type (RBPT ELISA MRT)').classes_))
        retained = st.selectbox("Retained Placenta", options=list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)').classes_))
        disposal = st.selectbox("Disposal", options=list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)').classes_))

    input_data = {
        'Age': age, 'Breed species': breed, 'Sex': sex, 'Calvings': calvings,
        'Abortion History (Yes No)': abortion, 'Infertility Repeat breeder(Yes No)': infertility,
        'Brucella vaccination status (Yes No)': vaccine, 'Sample Type(Serum Milk)': sample,
        'Test Type (RBPT ELISA MRT)': test_type, 'Retained Placenta Stillbirth(Yes No No Data)': retained,
        'Proper Disposal of Aborted Fetuses (Yes No)': disposal
    }

    if st.button(t["predict_btn"]):
        # 1. FIXED PRE-PROCESSING
        input_df = pd.DataFrame([input_data])
        
        # Robust Encoding
        for col in input_df.columns:
            if col in le_dict:
                input_df[col] = le_dict[col].transform(input_df[col].astype(str))
        
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scaling
        is_scaled_model = isinstance(best_model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))
        processed = scaler.transform(input_df) if is_scaled_model else input_df.values

        try:
            pred_idx = best_model.predict(processed)[0]
            probs = best_model.predict_proba(processed)[0]
            res_label = le_target.inverse_transform([pred_idx])[0]
            conf = probs.max()

            st.subheader(t["results_header"])
            st.success(f"Status: {res_label} ({conf:.2%})")

            if ai_enabled:
                with st.spinner(t["ai_loading"]):
                    prompt = t["system_prompt"].format(json.dumps(input_data), res_label, round(conf*100, 2))
                    response = gemini_model.generate_content(prompt)
                    st.info(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
