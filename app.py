import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from passlib.hash import pbkdf2_sha256 

warnings.filterwarnings('ignore')

# --- GEMINI CONFIGURATION ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    st.sidebar.warning("⚠️ Gemini API Key not found in secrets. AI advice will be disabled.")

# --- TRANSLATIONS & AUTOMATED PROMPT TEMPLATES ---
translations = {
    "English": {
        "welcome": "Welcome to Brucellosis Prediction App",
        "title": "🐂 Brucellosis Prediction Model",
        "user_greet": "Welcome, **{}**!",
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
        "results_header": "Prediction Results:",
        "pred_res": "**Predicted Status:**",
        "conf": "**Confidence Score:**",
        "prob_header": "Probability Analysis:",
        "chart_title": "Class Distribution",
        "logout": "Logout",
        "login_sub": "Login",
        "ai_advice_header": "🤖 AI Veterinary Consultation (Auto-Generated)",
        "ai_loading": "Analyzing data and generating suggestions...",
        # Hidden Prompt Template
        "system_prompt": "You are a senior veterinary expert. Analyzing animal data: {}. Prediction Result: {}. Confidence: {}. If result is Positive, strongly advise immediate isolation and confirmatory lab testing (RBPT/ELISA). Provide 3-4 clear, actionable steps for the farmer in English."
    },
    "Hindi": {
        "welcome": "ब्रुसेलोसिस भविष्यवाणी ऐप में आपका स्वागत है",
        "title": "🐂 ब्रुसेलोसिस भविष्यवाणी मॉडल",
        "user_greet": "आपका स्वागत है, **{}**!",
        "input_header": "इनपुट विशेषताएं",
        "age": "आयु (वर्ष)",
        "breed": "नस्ल/प्रजाति",
        "sex": "लिंग",
        "calvings": "बछड़े की संख्या",
        "abortion": "गर्भपात का इतिहास",
        "infertility": "बांझपन",
        "vaccination": "टीकाकरण की स्थिति",
        "sample": "नमूना प्रकार",
        "test": "परीक्षण प्रकार",
        "retained": "जेर रुकना/मृत प्रसव",
        "disposal": "भ्रूण का निपटान",
        "predict_btn": "भविष्यवाणी करें",
        "results_header": "परिणाम:",
        "pred_res": "**अनुमानित स्थिति:**",
        "conf": "**भरोसा (Confidence):**",
        "prob_header": "संभावना विश्लेषण:",
        "chart_title": "संभावना चार्ट",
        "logout": "लॉगआउट",
        "login_sub": "लॉगिन",
        "ai_advice_header": "🤖 AI पशु चिकित्सक सलाह (स्वचालित)",
        "ai_loading": "डेटा का विश्लेषण और सुझाव तैयार किए जा रहे हैं...",
        # Hidden Prompt Template
        "system_prompt": "आप एक वरिष्ठ पशु चिकित्सा विशेषज्ञ हैं। पशु डेटा: {}. भविष्यवाणी परिणाम: {}. भरोसा: {}%. यदि परिणाम पॉजिटिव है, तो तुरंत पशु को अलग करने (Isolation) और लैब टेस्टिंग (RBPT/ELISA) की सलाह दें। किसान के लिए हिंदी में 3-4 स्पष्ट और व्यावहारिक सुझाव दें।"
    }
}

# --- MODEL LOADING ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

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

# --- UI LOGIC ---
st.set_page_config(page_title="Brucella AI Predictor", layout="wide")
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
t = translations[selected_lang]

if not st.session_state['logged_in']:
    st.title(t["welcome"])
    st.sidebar.subheader(t["login_sub"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
        if email in users and pbkdf2_sha256.verify(password, users[email]):
            st.session_state['logged_in'] = True
            st.session_state['username'] = email
            st.rerun()
        else: st.sidebar.error("Invalid credentials")
else:
    st.title(t["title"])
    st.markdown(t["user_greet"].format(st.session_state['username']))
    st.sidebar.button(t["logout"], on_click=lambda: st.session_state.update(logged_in=False, username=None))

    # Features UI
    st.sidebar.header(t["input_header"])
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider(t["age"], 0, 20, 5)
        breed = st.selectbox(t["breed"], options=sorted(list(le_dict.get('Breed species').classes_)))
        sex = st.selectbox(t["sex"], options=sorted(list(le_dict.get('Sex').classes_)))
        calvings = st.slider(t["calvings"], 0, 15, 1)
        abortion = st.selectbox(t["abortion"], options=sorted(list(le_dict.get('Abortion History (Yes No)').classes_)))

    with col2:
        infertility = st.selectbox(t["infertility"], options=sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)').classes_)))
        vaccine = st.selectbox(t["vaccination"], options=sorted(list(le_dict.get('Brucella vaccination status (Yes No)').classes_)))
        sample = st.selectbox(t["sample"], options=sorted(list(le_dict.get('Sample Type(Serum Milk)').classes_)))
        test = st.selectbox(t["test"], options=sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)').classes_)))
        retained = st.selectbox(t["retained"], options=sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)').classes_)))
        disposal = st.selectbox(t["disposal"], options=sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)').classes_)))

    input_data = {
        'Age': age, 'Breed species': breed, 'Sex': sex, 'Calvings': calvings,
        'Abortion History (Yes No)': abortion, 'Infertility Repeat breeder(Yes No)': infertility,
        'Brucella vaccination status (Yes No)': vaccine, 'Sample Type(Serum Milk)': sample,
        'Test Type (RBPT ELISA MRT)': test, 'Retained Placenta Stillbirth(Yes No No Data)': retained,
        'Proper Disposal of Aborted Fetuses (Yes No)': disposal
    }

    if st.button(t["predict_btn"]):
        # 1. Prediction Logic
        # Create a dataframe with the exact feature names the model was trained on
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns are in the EXACT order the model expects
        input_df = input_df.reindex(columns=feature_names)

        # Handle Categorical Encoding manually to ensure no mismatch
        for col in input_df.columns:
            if col in le_dict:
                try:
                    # Transform the string input into the numeric value the model needs
                    input_df[col] = le_dict[col].transform(input_df[col])
                except Exception as e:
                    st.error(f"Error encoding {col}: {e}")
                    st.stop()

        # Decide if scaling is needed
        scaling = isinstance(best_model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))
        processed = scaler.transform(input_df) if scaling else input_df.values
        
        try:
            pred_idx = best_model.predict(processed)[0]
            probs = best_model.predict_proba(processed)[0]
            res_label = le_target.inverse_transform([pred_idx])[0]
            conf_score = probs.max()

            # Display Results
            st.markdown("---")
            st.subheader(t["results_header"])
            ui_res = "पॉजिटिव (Positive)" if (selected_lang == "Hindi" and "Positive" in res_label) else \
                     "नेगेटिव (Negative)" if (selected_lang == "Hindi" and "Negative" in res_label) else res_label
            
            st.success(f"{t['pred_res']} {ui_res}")
            st.info(f"{t['conf']} {conf_score:.2%}")

            # AUTOMATED AI ADVICE
            if "GEMINI_API_KEY" in st.secrets:
                st.subheader(t["ai_advice_header"])
                with st.spinner(t["ai_loading"]):
                    auto_prompt = t["system_prompt"].format(json.dumps(input_data), res_label, round(conf_score*100, 2))
                    response = gemini_model.generate_content(auto_prompt)
                    st.markdown(f"> {response.text}")
        
        except Exception as e:
            st.error(f"Model Error: {e}. Please check if input features match the model requirements.")

        # Visuals
        st.write("---")
        prob_df = pd.DataFrame({'Probability': probs}, index=le_target.classes_)
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(x=prob_df.index, y=prob_df['Probability'], palette='viridis', ax=ax)
        st.pyplot(fig)

