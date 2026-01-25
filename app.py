import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from passlib.hash import pbkdf2_sha256
import gspread
from google.oauth2.service_account import Credentials
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import time
import uuid

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="BrucellosisAI - Prediction System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GEMINI CONFIGURATION ---
ai_enabled = False
gemini_model = None

if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if available_models:
            gemini_model = genai.GenerativeModel(model_name=available_models[0])
            ai_enabled = True
    except: pass

# --- TRANSLATIONS ---
translations = {
    "English": {
        "dashboard": "Brucellosis Prediction Dashboard",
        "subtitle": "AI-powered disease prediction and veterinary consultation",
        "total_predictions": "Total Predictions",
        "positive_cases": "Positive Cases",
        "accuracy_rate": "Accuracy Rate",
        "ai_consultations": "AI Consultations",
        "input_header": "Animal Information Input",
        "input_subtitle": "Enter details for brucellosis prediction",
        "age": "Age (Years)", "breed": "Breed/Species", "sex": "Sex",
        "calvings": "Number of Calvings", "abortion": "Abortion History",
        "vaccination": "Vaccination Status", "sample": "Sample Type",
        "test": "Test Type", "retained": "Retained Placenta/Stillbirth",
        "disposal": "Proper Disposal of Aborted Fetuses", "infertility": "Infertility/Repeat Breeder",
        "run_prediction": "Run AI Prediction", "prediction_results": "Prediction Results",
        "probability_dist": "Probability Distribution", "run_prediction_msg": "Run a prediction to see results",
        "vet_assistant": "Veterinary AI Assistant", "vet_subtitle": "Get instant expert advice",
        "start_consultation": "Start Consultation", "quick_actions": "Quick Actions",
        "export_report": "Export Report", "schedule_test": "Schedule Test",
        "view_guidelines": "View Guidelines", "new_prediction": "New Prediction",
        "history": "History", "analytics": "Analytics", "ai_assistant": "AI Assistant",
        "guidelines": "Guidelines", "settings": "Settings", "logout": "Logout",
        "ai_insights": "AI-Powered Insights", "dashboard_menu": "Dashboard",
        "resources": "RESOURCES", "save_template": "Save Template",
        "predicted_status": "Predicted Status:", "confidence": "Confidence Score:"
    },
    "Hindi": {
        "dashboard": "ब्रुसेलोसिस भविष्यवाणी डैशबोर्ड",
        "subtitle": "AI-संचालित रोग भविष्यवाणी और पशु चिकित्सा परामर्श",
        "total_predictions": "कुल भविष्यवाणियां",
        "positive_cases": "पॉजिटिव केस",
        "accuracy_rate": "सटीकता दर",
        "ai_consultations": "AI परामर्श",
        "input_header": "पशु जानकारी इनपुट",
        "input_subtitle": "ब्रुसेलोसिस भविष्यवाणी के लिए विवरण दर्ज करें",
        "age": "आयु (वर्ष)", "breed": "नस्ल/प्रजाति", "sex": "लिंग",
        "calvings": "बछड़ों की संख्या", "abortion": "गर्भपात का इतिहास",
        "vaccination": "टीकाकरण स्थिति", "sample": "नमूना प्रकार",
        "test": "परीक्षण प्रकार", "retained": "जेर रुकना/मृत प्रसव",
        "disposal": "भ्रूण का निपटान", "infertility": "बांझपन",
        "run_prediction": "AI भविष्यवाणी चलाएं", "prediction_results": "भविष्यवाणी परिणाम",
        "probability_dist": "संभावना वितरण", "run_prediction_msg": "परिणाम देखने के लिए भविष्यवाणी चलाएं",
        "vet_assistant": "पशु चिकित्सा AI सहायक", "vet_subtitle": "तुरंत विशेषज्ञ सलाह प्राप्त करें",
        "start_consultation": "परामर्श शुरू करें", "quick_actions": "त्वरित क्रियाएं",
        "export_report": "रिपोर्ट निर्यात करें", "schedule_test": "परीक्षण निर्धारित करें",
        "view_guidelines": "दिशानिर्देश देखें", "new_prediction": "नई भविष्यवाणी",
        "history": "इतिहास", "analytics": "विश्लेषण", "ai_assistant": "AI सहायक",
        "guidelines": "दिशानिर्देश", "settings": "सेटिंग्स", "logout": "लॉगआउट",
        "ai_insights": "AI-संचालित अंतर्दष्टि", "dashboard_menu": "डैशबोर्ड",
        "resources": "संसाधन", "save_template": "टेम्पलेट सहेजें",
        "predicted_status": "अनुमानित स्थिति:", "confidence": "भरोसा:"
    }
}

# --- SESSION STATE & PERSISTENCE ---
@st.cache_resource
def get_session_cache(): return {}
session_cache = get_session_cache()

def init_session():
    try:
        query_params = st.query_params
        session_id = query_params.get("session_id", None)
    except:
        try:
            query_params = st.experimental_get_query_params()
            session_id = query_params.get("session_id", [None])[0]
        except: session_id = None

    if session_id and session_id in session_cache:
        user_data = session_cache[session_id]
        if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
            st.session_state['logged_in'] = True
            st.session_state['username'] = user_data['username']
            st.session_state['current_session_id'] = session_id
    
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
    if 'show_chatbot' not in st.session_state: st.session_state['show_chatbot'] = False
    if 'prediction_count' not in st.session_state: st.session_state['prediction_count'] = 1247
    if 'positive_count' not in st.session_state: st.session_state['positive_count'] = 87
    if 'ai_consultation_count' not in st.session_state: st.session_state['ai_consultation_count'] = 342
    if 'last_prediction' not in st.session_state: st.session_state['last_prediction'] = None
    if 'otp_sent' not in st.session_state: st.session_state['otp_sent'] = False
    if 'form_data' not in st.session_state:
        st.session_state['form_data'] = {'age': 5, 'breed': None, 'sex': None, 'calvings': 1, 'abortion': None, 'infertility': None, 'vaccine': None, 'sample': None, 'test': None, 'retained': None, 'disposal': None}

init_session()

# --- CHAT CALLBACKS (FIXED FOR NO RELOAD) ---
def handle_chat_submit():
    if st.session_state.chat_input:
        user_input = st.session_state.chat_input
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        try:
            lang = st.session_state.get('selected_language', 'English')
            prompt = f"You are a veterinary consultant specializing in Brucellosis. Answer in {lang}: {user_input}"
            response = gemini_model.generate_content(prompt)
            st.session_state['chat_history'].append({"role": "assistant", "content": response.text})
            st.session_state['ai_consultation_count'] += 1
        except Exception as e: st.session_state['chat_error'] = str(e)

def handle_chat_clear(): st.session_state['chat_history'] = []

# --- CORE FUNCTIONS (OTP, SHEETS, ARTIFACTS) ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'
GOOGLE_SHEET_ID = '159z65oDmaBPymwndIHkNVbK1Q6_GMmFc7xGcJ2fsozY'

@st.cache_resource
def load_all_artifacts():
    try:
        with open(MODEL_ARTIFACTS_DIR + 'best_model.pkl', 'rb') as f: m = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'le_dict.pkl', 'rb') as f: ld = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'le_target.pkl', 'rb') as f: lt = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'scaler.pkl', 'rb') as f: s = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'feature_names.pkl', 'rb') as f: fn = pickle.load(f)
        return m, ld, lt, s, fn
    except: return None, None, None, None, None

best_model, le_dict, le_target, scaler, feature_names = load_all_artifacts()

# --- UI LOGIC ---
if not st.session_state['logged_in']:
    st.title("BrucellosisAI Login")
    with st.form("login"):
        e = st.text_input("Email")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Access Dashboard"):
            st.session_state.update(logged_in=True, username=e)
            st.rerun()
else:
    selected_lang = st.sidebar.selectbox("🌐 Language", ["English", "Hindi"], key="selected_language")
    t = translations[selected_lang]
    
    with st.sidebar:
        st.title("BrucellosisAI")
        if st.button(f"📊 {t['dashboard_menu']}"): st.session_state['current_page'] = 'dashboard'
        if ai_enabled and st.button(f"🤖 {t['ai_assistant']}"):
            st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
        if st.button(f"🚪 {t['logout']}", type="primary"): 
            st.session_state.clear()
            st.rerun()

    st.title(t["dashboard"])
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"📝 {t['input_header']}")
        with st.form("prediction_form"):
            ca, cb = st.columns(2)
            with ca:
                age = st.number_input(t["age"], 0, 20, 5)
                breed = st.selectbox(t["breed"], options=sorted(list(le_dict['Breed species'].classes_)) if le_dict else ["..."])
                sex = st.selectbox(t["sex"], options=sorted(list(le_dict['Sex'].classes_)) if le_dict else ["..."])
                calvings = st.number_input(t["calvings"], 0, 15, 1)
                abortion = st.selectbox(t["abortion"], options=sorted(list(le_dict['Abortion History (Yes No)'].classes_)) if le_dict else ["..."])
            with cb:
                infertility = st.selectbox(t["infertility"], options=sorted(list(le_dict['Infertility Repeat breeder(Yes No)'].classes_)) if le_dict else ["..."])
                vaccine = st.selectbox(t["vaccination"], options=sorted(list(le_dict['Brucella vaccination status (Yes No)'].classes_)) if le_dict else ["..."])
                sample = st.selectbox(t["sample"], options=sorted(list(le_dict['Sample Type(Serum Milk)'].classes_)) if le_dict else ["..."])
                test = st.selectbox(t["test"], options=sorted(list(le_dict['Test Type (RBPT ELISA MRT)'].classes_)) if le_dict else ["..."])
                retained = st.selectbox(t["retained"], options=sorted(list(le_dict['Retained Placenta Stillbirth(Yes No No Data)'].classes_)) if le_dict else ["..."])
            
            if st.form_submit_button(f"🔬 {t['run_prediction']}", type="primary"):
                # Data cleaning and Prediction
                input_data = {'Age': age, 'Breed species': breed, 'Sex': sex, 'Calvings': calvings, 'Abortion History (Yes No)': abortion, 'Infertility Repeat breeder(Yes No)': infertility, 'Brucella vaccination status (Yes No)': vaccine, 'Sample Type(Serum Milk)': sample, 'Test Type (RBPT ELISA MRT)': test, 'Retained Placenta Stillbirth(Yes No No Data)': retained, 'Proper Disposal of Aborted Fetuses (Yes No)': 'No Data'}
                df = pd.DataFrame([input_data])
                
                try:
                    for col in df.columns:
                        if col in le_dict:
                            val = str(df[col].iloc[0]).strip()
                            df[col] = le_dict[col].transform([val]) if val in le_dict[col].classes_ else 0
                    
                    df = df.reindex(columns=feature_names, fill_value=0)
                    probs = best_model.predict_proba(df)[0]
                    st.session_state['last_prediction'] = {'result': le_target.inverse_transform([best_model.predict(df)[0]])[0], 'confidence': probs.max(), 'probabilities': probs, 'classes': le_target.classes_}
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")

        if st.session_state.get('last_prediction'):
            p = st.session_state['last_prediction']
            st.info(f"### {p['result']} ({p['confidence']:.1%})")

    with col_right:
        if ai_enabled:
            st.subheader(f"🤖 {t['vet_assistant']}")
            if st.button(f"💬 {t['start_consultation']}", use_container_width=True):
                st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
            
            if st.session_state['show_chatbot']:
                chat_box = st.container(height=400)
                with chat_box:
                    for msg in st.session_state['chat_history']:
                        with st.chat_message(msg["role"]): st.write(msg["content"])
                
                with st.form("chat_form", clear_on_submit=True):
                    st.text_input("💬 Your question...", key="chat_input")
                    st.form_submit_button("Send", on_click=handle_chat_submit)

st.markdown("---")
st.caption("Developed by LNMIIT Student Internship")
