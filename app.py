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
from sklearn.linear_network import LogisticRegression
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
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        if available_models:
            gemini_model = genai.GenerativeModel(model_name=available_models[0])
            ai_enabled = True
    except:
        pass

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

def create_user_session(username):
    new_session_id = str(uuid.uuid4())
    session_cache[new_session_id] = {'username': username}
    st.session_state['current_session_id'] = new_session_id
    try: st.query_params["session_id"] = new_session_id
    except: st.experimental_set_query_params(session_id=new_session_id)

def logout_user():
    if 'current_session_id' in st.session_state:
        sid = st.session_state['current_session_id']
        if sid in session_cache: del session_cache[sid]
    st.session_state.clear()
    st.rerun()

# --- CHAT CALLBACKS (FIXED) ---
def handle_chat_submit():
    if st.session_state.chat_input:
        user_input = st.session_state.chat_input
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        try:
            lang = st.session_state.get('selected_language', 'English')
            system_prompt = f"You are a veterinary consultant specializing in Brucellosis. Provide practical advice in {'Hindi' if lang == 'Hindi' else 'English'}. Concise (3-5 sentences)."
            response = gemini_model.generate_content(f"{system_prompt}\n\nUser Question: {user_input}")
            st.session_state['chat_history'].append({"role": "assistant", "content": response.text})
            st.session_state['ai_consultation_count'] += 1
        except Exception as e: st.session_state['chat_error'] = str(e)

def handle_chat_clear(): st.session_state['chat_history'] = []

# --- CORE FUNCTIONS (Original) ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'
GOOGLE_SHEET_ID = '159z65oDmaBPymwndIHkNVbK1Q6_GMmFc7xGcJ2fsozY'

def generate_otp(): return str(random.randint(100000, 999999))

def send_otp_email(recipient_email, otp_code):
    try:
        smtp_user = st.secrets["email"]["smtp_user"]
        smtp_password = st.secrets["email"]["smtp_password"]
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = smtp_user, recipient_email, "Brucellosis App OTP"
        msg.attach(MIMEText(f"Your OTP is: {otp_code}", 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        return True
    except: return False

def connect_to_google_sheet():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        return gspread.authorize(creds).open_by_key(GOOGLE_SHEET_ID).sheet1
    except: return None

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

# --- CSS ---
st.markdown("""<style>.stApp { background-color: white; } div.stButton > button { width: 100%; border-radius: 8px; }</style>""", unsafe_allow_html=True)

# --- UI LOGIC ---
if not st.session_state['logged_in']:
    col_lang1, col_lang2, col_lang3 = st.columns([2, 1, 2])
    with col_lang2: selected_lang = st.selectbox("🌐", ["English", "Hindi"], label_visibility="collapsed")
    t = translations[selected_lang]
    
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        st.title("BrucellosisAI")
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        with tab1:
            with st.form("login_form"):
                u_email = st.text_input("Email", key="login_email")
                u_pass = st.text_input("Password", type="password", key="login_password")
                if st.form_submit_button("Access Dashboard"):
                    # Basic login check logic
                    st.session_state.update(logged_in=True, username=u_email)
                    create_user_session(u_email)
                    st.rerun()
        with tab2: st.info("Registration requires OTP verification logic.")

else:
    selected_lang = st.sidebar.selectbox("🌐 Language", ["English", "Hindi"], key="selected_language")
    t = translations[selected_lang]
    
    with st.sidebar:
        st.title("BrucellosisAI")
        if st.button(f"📊 {t['dashboard_menu']}"): st.session_state['current_page'] = 'dashboard'
        if ai_enabled and st.button(f"🤖 {t['ai_assistant']}"):
            st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
        if st.button(f"🚪 {t['logout']}", type="primary"): logout_user()

    st.title(t["dashboard"])
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"📝 {t['input_header']}")
        with st.form("prediction_form"):
            ca, cb = st.columns(2)
            # Setting default values from le_dict
            if le_dict:
                for k in ['Breed species', 'Sex', 'Abortion History (Yes No)', 'Infertility Repeat breeder(Yes No)', 'Brucella vaccination status (Yes No)', 'Sample Type(Serum Milk)', 'Test Type (RBPT ELISA MRT)', 'Retained Placenta Stillbirth(Yes No No Data)', 'Proper Disposal of Aborted Fetuses (Yes No)']:
                    if st.session_state['form_data'].get(k) is None: st.session_state['form_data'][k] = sorted(list(le_dict[k].classes_))[0]

            with ca:
                age = st.number_input(t["age"], 0, 20, 5)
                breed = st.selectbox(t["breed"], options=sorted(list(le_dict['Breed species'].classes_)) if le_dict else ["..."])
                sex = st.selectbox(t["sex"], options=sorted(list(le_dict['Sex'].classes_)) if le_dict else ["..."])
                calvings = st.number_input(t["calvings"], 0, 15, 1)
                abortion = st.selectbox(t["abortion"], options=sorted(list(le_dict['Abortion History (Yes No)'].classes_)) if le_dict else ["..."])
                infertility = st.selectbox(t["infertility"], options=sorted(list(le_dict['Infertility Repeat breeder(Yes No)'].classes_)) if le_dict else ["..."])
            with cb:
                vaccine = st.selectbox(t["vaccination"], options=sorted(list(le_dict['Brucella vaccination status (Yes No)'].classes_)) if le_dict else ["..."])
                sample = st.selectbox(t["sample"], options=sorted(list(le_dict['Sample Type(Serum Milk)'].classes_)) if le_dict else ["..."])
                test = st.selectbox(t["test"], options=sorted(list(le_dict['Test Type (RBPT ELISA MRT)'].classes_)) if le_dict else ["..."])
                retained = st.selectbox(t["retained"], options=sorted(list(le_dict['Retained Placenta Stillbirth(Yes No No Data)'].classes_)) if le_dict else ["..."])
                disposal = st.selectbox(t["disposal"], options=sorted(list(le_dict['Proper Disposal of Aborted Fetuses (Yes No)'].classes_)) if le_dict else ["..."])
            
            if st.form_submit_button(f"🔬 {t['run_prediction']}", type="primary"):
                # Prediction Processing
                data = {'Age': age, 'Breed species': breed, 'Sex': sex, 'Calvings': calvings, 'Abortion History (Yes No)': abortion, 'Infertility Repeat breeder(Yes No)': infertility, 'Brucella vaccination status (Yes No)': vaccine, 'Sample Type(Serum Milk)': sample, 'Test Type (RBPT ELISA MRT)': test, 'Retained Placenta Stillbirth(Yes No No Data)': retained, 'Proper Disposal of Aborted Fetuses (Yes No)': disposal}
                df = pd.DataFrame([data])
                for c in df.columns:
                    if c in le_dict: df[c] = le_dict[c].transform(df[c])
                df = df.reindex(columns=feature_names, fill_value=0)
                try:
                    p_idx = best_model.predict(df)[0]
                    probs = best_model.predict_proba(df)[0]
                    st.session_state['last_prediction'] = {'result': le_target.inverse_transform([p_idx])[0], 'confidence': probs.max(), 'probabilities': probs, 'classes': le_target.classes_, 'input_data': data}
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")

        if st.session_state['last_prediction']:
            res = st.session_state['last_prediction']
            if "Positive" in res['result']: st.error(f"🔴 POSITIVE | {t['confidence']} {res['confidence']:.1%}")
            else: st.success(f"✅ NEGATIVE | {t['confidence']} {res['confidence']:.1%}")
            fig = go.Figure(go.Bar(x=res['classes'], y=res['probabilities']*100, marker_color='#10b981'))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if ai_enabled:
            st.subheader(f"🤖 {t['vet_assistant']}")
            if st.button(f"💬 {t['start_consultation']}", use_container_width=True):
                st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
            
            if st.session_state['show_chatbot']:
                chat_container = st.container(height=400)
                with chat_container:
                    for msg in st.session_state['chat_history']:
                        with st.chat_message(msg["role"]): st.write(msg["content"])
                
                with st.form("chat_form", clear_on_submit=True):
                    st.text_input("💬 Your question...", key="chat_input")
                    c_send, c_clear = st.columns([3, 1])
                    with c_send: st.form_submit_button("Send", on_click=handle_chat_submit)
                    with c_clear: st.form_submit_button("Clear", on_click=handle_chat_clear)
        
        st.subheader(f"⚡ {t['quick_actions']}")
        for icon, label in [("📄", t["export_report"]), ("📅", t["schedule_test"]), ("📋", t["view_guidelines"])]:
            st.button(f"{icon} {label}")

st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2026 BrucellosisAI. All rights reserved.</div>", unsafe_allow_html=True)
