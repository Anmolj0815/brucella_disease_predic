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
        "age": "Age (Years)",
        "breed": "Breed/Species",
        "sex": "Sex",
        "calvings": "Number of Calvings",
        "abortion": "Abortion History",
        "vaccination": "Vaccination Status",
        "sample": "Sample Type",
        "test": "Test Type",
        "retained": "Retained Placenta/Stillbirth",
        "disposal": "Proper Disposal of Aborted Fetuses",
        "infertility": "Infertility/Repeat Breeder",
        "run_prediction": "Run AI Prediction",
        "prediction_results": "Prediction Results",
        "probability_dist": "Probability Distribution",
        "run_prediction_msg": "Run a prediction to see results",
        "vet_assistant": "Veterinary AI Assistant",
        "vet_subtitle": "Get instant expert advice on brucellosis, milk safety, and animal health",
        "start_consultation": "Start Consultation",
        "quick_actions": "Quick Actions",
        "export_report": "Export Report",
        "schedule_test": "Schedule Test",
        "view_guidelines": "View Guidelines",
        "new_prediction": "New Prediction",
        "history": "History",
        "analytics": "Analytics",
        "ai_assistant": "AI Assistant",
        "guidelines": "Guidelines",
        "settings": "Settings",
        "logout": "Logout",
        "ai_insights": "AI-Powered Insights",
        "dashboard_menu": "Dashboard",
        "resources": "RESOURCES",
        "save_template": "Save Template",
        "predicted_status": "Predicted Status:",
        "confidence": "Confidence Score:"
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
        "age": "आयु (वर्ष)",
        "breed": "नस्ल/प्रजाति",
        "sex": "लिंग",
        "calvings": "बछड़ों की संख्या",
        "abortion": "गर्भपात का इतिहास",
        "vaccination": "टीकाकरण स्थिति",
        "sample": "नमूना प्रकार",
        "test": "परीक्षण प्रकार",
        "retained": "जेर रुकना/मृत प्रसव",
        "disposal": "भ्रूण का निपटान",
        "infertility": "बांझपन",
        "run_prediction": "AI भविष्यवाणी चलाएं",
        "prediction_results": "भविष्यवाणी परिणाम",
        "probability_dist": "संभावना वितरण",
        "run_prediction_msg": "परिणाम देखने के लिए भविष्यवाणी चलाएं",
        "vet_assistant": "पशु चिकित्सा AI सहायक",
        "vet_subtitle": "ब्रुसेलोसिस, दूध सुरक्षा और पशु स्वास्थ्य पर तुरंत विशेषज्ञ सलाह प्राप्त करें",
        "start_consultation": "परामर्श शुरू करें",
        "quick_actions": "त्वरित क्रियाएं",
        "export_report": "रिपोर्ट निर्यात करें",
        "schedule_test": "परीक्षण निर्धारित करें",
        "view_guidelines": "दिशानिर्देश देखें",
        "new_prediction": "नई भविष्यवाणी",
        "history": "इतिहास",
        "analytics": "विश्लेषण",
        "ai_assistant": "AI सहायक",
        "guidelines": "दिशानिर्देश",
        "settings": "सेटिंग्स",
        "logout": "लॉगआउट",
        "ai_insights": "AI-संचालित अंतर्दृष्टि",
        "dashboard_menu": "डैशबोर्ड",
        "resources": "संसाधन",
        "save_template": "टेम्पलेट सहेजें",
        "predicted_status": "अनुमानित स्थिति:",
        "confidence": "भरोसा:"
    }
}

# --- SESSION STATE & PERSISTENCE ---
import uuid

@st.cache_resource
def get_session_cache():
    return {}

session_cache = get_session_cache()

def init_session():
    # Check for query param session_id using appropriate Streamlit version method
    try:
        # For newer Streamlit versions
        query_params = st.query_params
        session_id = query_params.get("session_id", None)
    except:
        # Fallback for older Streamlit versions (like 1.29.0)
        try:
            query_params = st.experimental_get_query_params()
            session_id = query_params.get("session_id", [None])[0]
        except:
            session_id = None

    if session_id and session_id in session_cache:
        user_data = session_cache[session_id]
        if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
            st.session_state['logged_in'] = True
            st.session_state['username'] = user_data['username']
            st.session_state['current_session_id'] = session_id
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'show_chatbot' not in st.session_state:
        st.session_state['show_chatbot'] = False
    if 'prediction_count' not in st.session_state:
        st.session_state['prediction_count'] = 1247
    if 'positive_count' not in st.session_state:
        st.session_state['positive_count'] = 87
    if 'ai_consultation_count' not in st.session_state:
        st.session_state['ai_consultation_count'] = 342
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None
    if 'otp_sent' not in st.session_state:
        st.session_state['otp_sent'] = False
    if 'otp_code' not in st.session_state:
        st.session_state['otp_code'] = None
    if 'otp_timestamp' not in st.session_state:
        st.session_state['otp_timestamp'] = None
    if 'pending_user_data' not in st.session_state:
        st.session_state['pending_user_data'] = None
    if 'form_data' not in st.session_state:
        st.session_state['form_data'] = {
            'age': 5,
            'breed': None,
            'sex': None,
            'calvings': 1,
            'abortion': None,
            'infertility': None,
            'vaccine': None,
            'sample': None,
            'test': None,
            'retained': None,
            'disposal': None
        }

init_session()

def create_user_session(username):
    new_session_id = str(uuid.uuid4())
    session_cache[new_session_id] = {'username': username}
    st.session_state['current_session_id'] = new_session_id
    try:
        st.query_params["session_id"] = new_session_id
    except:
        st.experimental_set_query_params(session_id=new_session_id)

def logout_user():
    if 'current_session_id' in st.session_state:
        sid = st.session_state['current_session_id']
        if sid in session_cache:
            del session_cache[sid]
    st.session_state.clear()
    try:
        st.query_params.clear()
    except:
        st.experimental_set_query_params()
    st.rerun()

# --- MODEL LOADING ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'
GOOGLE_SHEET_ID = '159z65oDmaBPymwndIHkNVbK1Q6_GMmFc7xGcJ2fsozY'

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(recipient_email, otp_code):
    try:
        smtp_user = st.secrets["email"]["smtp_user"]
        smtp_password = st.secrets["email"]["smtp_password"]
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = recipient_email
        msg['Subject'] = "Brucellosis App - Email Verification OTP"
        body = f"""Hello,\n\nYour OTP for Brucellosis Prediction App registration is: {otp_code}\n\nThis OTP is valid for 10 minutes.\n\nBest regards,\nBrucellosis App Team"""
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")
        return False

def verify_otp(entered_otp):
    if st.session_state['otp_code'] is None:
        return False, "No OTP sent"
    if time.time() - st.session_state['otp_timestamp'] > 600:
        return False, "OTP expired. Please request a new one."
    if entered_otp == st.session_state['otp_code']:
        return True, "OTP verified successfully"
    else:
        return False, "Invalid OTP. Please try again."

def connect_to_google_sheet():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1
        return sheet
    except Exception as e:
        st.error(f"Google Sheets connection error: {e}")
        return None

def save_user_to_google_sheet(email, name, phone, location):
    try:
        sheet = connect_to_google_sheet()
        if sheet is None:
            return False
        try:
            headers = sheet.row_values(1)
            if not headers:
                sheet.append_row(['Email', 'Name', 'Phone', 'Location', 'Registration Date'])
        except:
            sheet.append_row(['Email', 'Name', 'Phone', 'Location', 'Registration Date'])
        registration_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([email, name, phone, location, registration_date])
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheet: {e}")
        return False

def register_user(email, password, name, phone, location):
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
        else:
            users = {}
        if email in users:
            return False, "User already exists"
        users[email] = pbkdf2_sha256.hash(password)
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
        if save_user_to_google_sheet(email, name, phone, location):
            return True, "Registration successful"
        else:
            return False, "User created but Google Sheet save failed"
    except Exception as e:
        return False, f"Registration error: {e}"

@st.cache_resource
def load_all_artifacts():
    try:
        with open(MODEL_ARTIFACTS_DIR + 'best_model.pkl', 'rb') as f: m = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'le_dict.pkl', 'rb') as f: ld = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'le_target.pkl', 'rb') as f: lt = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'scaler.pkl', 'rb') as f: s = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'feature_names.pkl', 'rb') as f: fn = pickle.load(f)
        return m, ld, lt, s, fn
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None, None

best_model, le_dict, le_target, scaler, feature_names = load_all_artifacts()

# --- ENHANCED CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #4f46e5;
        --secondary: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #0f172a;
        --light: #f8fafc;
        --glass-bg: rgba(255, 255, 255, 0.9);
        --glass-border: rgba(255, 255, 255, 0.2);
    }

    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Dynamic Animated Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism Containers */
    .main .block-container {
        padding-top: 2rem;
        max-width: 95%;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    
    /* Modern Buttons in Sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-color: transparent;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }

    /* Cards */
    .dashboard-header, .metric-card, .section-card, .prediction-result {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .dashboard-header {
        padding: 2.5rem;
        background: rgba(255, 255, 255, 0.95);
        border-left: 8px solid var(--primary);
    }
    
    .metric-card:hover, .section-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric Cards Specifics */
    .metric-card {
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    /* Typography */
    .dashboard-title {
        background: linear-gradient(to right, #1e293b, #4b5563);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #1e293b, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Form Elements */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
        border: none;
        padding: 0.8rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        transform: translateY(-2px);
    }
    
    /* AI Chat Interface */
    .ai-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
    }
    
    .chat-container {
        border: none;
        background: #f8fafc;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .chat-message.user {
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 20px 20px 0 20px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    
    .chat-message.assistant {
        background: white;
        color: #1e293b;
        border: none;
        border-radius: 20px 20px 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* Login Page */
    .login-container-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }
    
    .login-box {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 3rem;
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.2);
        width: 100%;
        max-width: 450px;
        border: 1px solid white;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIN/REGISTER ---
if not st.session_state['logged_in']:
    # Language selector at top
    col_lang1, col_lang2, col_lang3 = st.columns([2, 1, 2])
    with col_lang2:
        selected_lang = st.selectbox("🌐", ["English", "Hindi"], label_visibility="collapsed")
    
    t = translations[selected_lang]
    
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        st.markdown("""
        <div class="login-box">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 0.5rem; animation: float 6s ease-in-out infinite;">🧬</div>
                <h1 style="color: #1e293b; font-weight: 800; margin-bottom: 0.5rem;">BrucellosisAI</h1>
                <p style="color: #64748b;">Next Gen Disease Prediction</p>
            </div>
            
            <style>
                @keyframes float {
                    0% { transform: translateY(0px); }
                    50% { transform: translateY(-10px); }
                    100% { transform: translateY(0px); }
                }
            </style>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                st.text_input("Email", key="login_email")
                st.text_input("Password", type="password", key="login_password")
                submit = st.form_submit_button("Access Dashboard", use_container_width=True, type="primary")
                
                if submit:
                    try:
                        with open(USERS_FILE, 'r') as f:
                            users = json.load(f)
                        if st.session_state.login_email in users and pbkdf2_sha256.verify(st.session_state.login_password, users[st.session_state.login_email]):
                            st.session_state.update(logged_in=True, username=st.session_state.login_email)
                            create_user_session(st.session_state.login_email)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    except:
                        st.error("❌ Database error")
        
        # Div moved to after tab2
        
        with tab2:
            if not st.session_state['otp_sent']:
                with st.form("register_form"):
                    st.text_input("👤 Full Name", key="reg_name")
                    st.text_input("📧 Email Address", key="reg_email")
                    st.text_input("📱 Phone Number", key="reg_phone")
                    st.text_input("📍 Location (City/Village)", key="reg_location")
                    st.text_input("🔒 Password", type="password", key="reg_password")
                    st.text_input("🔒 Confirm Password", type="password", key="reg_confirm")
                    submit_reg = st.form_submit_button("Send Verification Code", use_container_width=True, type="primary")
                    
                    if submit_reg:
                        if not all([st.session_state.reg_name, st.session_state.reg_email, st.session_state.reg_phone, st.session_state.reg_location, st.session_state.reg_password]):
                            st.error("❌ Please fill in all fields")
                        elif st.session_state.reg_password != st.session_state.reg_confirm:
                            st.error("❌ Passwords do not match")
                        elif len(st.session_state.reg_password) < 6:
                            st.error("❌ Password must be at least 6 characters")
                        else:
                            otp = generate_otp()
                            if send_otp_email(st.session_state.reg_email, otp):
                                st.session_state['otp_code'] = otp
                                st.session_state['otp_timestamp'] = time.time()
                                st.session_state['otp_sent'] = True
                                st.session_state['pending_user_data'] = {
                                    'email': st.session_state.reg_email,
                                    'password': st.session_state.reg_password,
                                    'name': st.session_state.reg_name,
                                    'phone': st.session_state.reg_phone,
                                    'location': st.session_state.reg_location
                                }
                                st.success(f"✅ Verification code sent to {st.session_state.reg_email}")
                                st.rerun()
            else:
                st.info(f"📧 Verification code sent to {st.session_state['pending_user_data']['email']}")
                entered_otp = st.text_input("🔢 Enter 6-digit Verification Code", max_chars=6)
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Verify Code", use_container_width=True, type="primary"):
                        if entered_otp:
                            is_valid, message = verify_otp(entered_otp)
                            if is_valid:
                                user_data = st.session_state['pending_user_data']
                                success, reg_message = register_user(user_data['email'], user_data['password'], user_data['name'], user_data['phone'], user_data['location'])
                                if success:
                                    st.success("✅ " + reg_message + " Please login now.")
                                    st.session_state['otp_sent'] = False
                                    st.session_state['otp_code'] = None
                                    st.session_state['otp_timestamp'] = None
                                    st.session_state['pending_user_data'] = None
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("❌ " + reg_message)
                            else:
                                st.error("❌ " + message)
                with col_b:
                    if st.button("🔄 Resend Code", use_container_width=True):
                        otp = generate_otp()
                        if send_otp_email(st.session_state['pending_user_data']['email'], otp):
                            st.session_state['otp_code'] = otp
                            st.session_state['otp_timestamp'] = time.time()
                            st.success("✅ New code sent")
                            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

else:
    # --- SIDEBAR MENU ---
    selected_lang = st.sidebar.selectbox("🌐 Language", ["English", "Hindi"], label_visibility="collapsed")
    t = translations[selected_lang]
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; border-bottom: 2px solid rgba(255,255,255,0.15); margin-bottom: 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔬</div>
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 700;">BrucellosisAI</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div style="color: rgba(255,255,255,0.6); font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; padding: 0 1rem; margin-bottom: 0.75rem;">MAIN MENU</div>', unsafe_allow_html=True)
        
        if st.button(f"📊 {t['dashboard_menu']}", use_container_width=True):
            st.session_state['current_page'] = 'dashboard'
        if st.button(f"➕ {t['new_prediction']}", use_container_width=True):
            st.session_state['current_page'] = 'new_prediction'
        if st.button(f"📜 {t['history']}", use_container_width=True):
            st.session_state['current_page'] = 'history'
        if st.button(f"📈 {t['analytics']}", use_container_width=True):
            st.session_state['current_page'] = 'analytics'
        
        st.markdown(f'<div style="color: rgba(255,255,255,0.6); font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; padding: 0 1rem; margin: 1.5rem 0 0.75rem 0;">{t["resources"]}</div>', unsafe_allow_html=True)
        
        if ai_enabled:
            if st.button(f"🤖 {t['ai_assistant']}", use_container_width=True, key="sidebar_ai_toggle"):
                st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
                st.rerun()
        
        if st.button(f"📋 {t['guidelines']}", use_container_width=True):
            pass
        if st.button(f"⚙️ {t['settings']}", use_container_width=True):
            pass
        
        st.markdown("<br>" * 4, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="padding: 1rem; background: rgba(255,255,255,0.15); border-radius: 12px; margin-top: auto; border: 1px solid rgba(255,255,255,0.2);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 40px; height: 40px; background: rgba(255,255,255,0.2); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">👤</div>
                <div>
                    <p style="color: white; margin: 0; font-size: 0.9rem; font-weight: 600;">Dr. User</p>
                    <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.75rem;">Veterinarian</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button(f"🚪 {t['logout']}", use_container_width=True, type="primary"):
            logout_user()
    
    # --- MAIN CONTENT ---
    # Dashboard Header
    st.markdown(f"""
    <div class="dashboard-header">
        <h1 class="dashboard-title">{t["dashboard"]}</h1>
        <p class="dashboard-subtitle">{t["subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-header">
                <div>
                    <div class="metric-label">{t["total_predictions"]}</div>
                    <div class="metric-value">{st.session_state['prediction_count']:,}</div>
                </div>
                <div class="metric-icon">📊</div>
            </div>
            <div class="metric-change positive">↑ +12.5% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card red">
            <div class="metric-header">
                <div>
                    <div class="metric-label">{t["positive_cases"]}</div>
                    <div class="metric-value">{st.session_state['positive_count']}</div>
                </div>
                <div class="metric-icon">⚠️</div>
            </div>
            <div class="metric-change negative">↑ +3.2% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-header">
                <div>
                    <div class="metric-label">{t["accuracy_rate"]}</div>
                    <div class="metric-value">94.3%</div>
                </div>
                <div class="metric-icon">✅</div>
            </div>
            <div class="metric-change positive">↑ +1.8% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="metric-header">
                <div>
                    <div class="metric-label">{t["ai_consultations"]}</div>
                    <div class="metric-value">{st.session_state['ai_consultation_count']}</div>
                </div>
                <div class="metric-icon">🤖</div>
            </div>
            <div class="metric-change positive">↑ +24.1% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Input Section
        st.markdown(f"""
        <div class="section-card">
            <div class="section-header">📝 {t["input_header"]}</div>
            <p class="section-subtitle">{t["input_subtitle"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Form inputs in a card-like container
        with st.container():
            col_a, col_b = st.columns(2)
            
            # Initialize form data with defaults from session state
            if st.session_state['form_data']['breed'] is None and le_dict:
                st.session_state['form_data']['breed'] = sorted(list(le_dict.get('Breed species').classes_))[0]
            if st.session_state['form_data']['sex'] is None and le_dict:
                st.session_state['form_data']['sex'] = sorted(list(le_dict.get('Sex').classes_))[0]
            if st.session_state['form_data']['abortion'] is None and le_dict:
                st.session_state['form_data']['abortion'] = sorted(list(le_dict.get('Abortion History (Yes No)').classes_))[0]
            if st.session_state['form_data']['infertility'] is None and le_dict:
                st.session_state['form_data']['infertility'] = sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)').classes_))[0]
            if st.session_state['form_data']['vaccine'] is None and le_dict:
                st.session_state['form_data']['vaccine'] = sorted(list(le_dict.get('Brucella vaccination status (Yes No)').classes_))[0]
            if st.session_state['form_data']['sample'] is None and le_dict:
                st.session_state['form_data']['sample'] = sorted(list(le_dict.get('Sample Type(Serum Milk)').classes_))[0]
            if st.session_state['form_data']['test'] is None and le_dict:
                st.session_state['form_data']['test'] = sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)').classes_))[0]
            if st.session_state['form_data']['retained'] is None and le_dict:
                st.session_state['form_data']['retained'] = sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)').classes_))[0]
            if st.session_state['form_data']['disposal'] is None and le_dict:
                st.session_state['form_data']['disposal'] = sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)').classes_))[0]
            
            with col_a:
                age = st.number_input(t["age"], min_value=0, max_value=20, value=st.session_state['form_data']['age'], step=1, key="age_input")
                breed = st.selectbox(t["breed"], options=sorted(list(le_dict.get('Breed species').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Breed species').classes_)).index(st.session_state['form_data']['breed']) if le_dict and st.session_state['form_data']['breed'] else 0, key="breed_input")
                sex = st.selectbox(t["sex"], options=sorted(list(le_dict.get('Sex').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Sex').classes_)).index(st.session_state['form_data']['sex']) if le_dict and st.session_state['form_data']['sex'] else 0, key="sex_input")
                calvings = st.number_input(t["calvings"], min_value=0, max_value=15, value=st.session_state['form_data']['calvings'], step=1, key="calvings_input")
                abortion = st.selectbox(t["abortion"], options=sorted(list(le_dict.get('Abortion History (Yes No)').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Abortion History (Yes No)').classes_)).index(st.session_state['form_data']['abortion']) if le_dict and st.session_state['form_data']['abortion'] else 0, key="abortion_input")
                infertility = st.selectbox(t["infertility"], options=sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)').classes_)).index(st.session_state['form_data']['infertility']) if le_dict and st.session_state['form_data']['infertility'] else 0, key="infertility_input")
            
            with col_b:
                vaccine = st.selectbox(t["vaccination"], options=sorted(list(le_dict.get('Brucella vaccination status (Yes No)').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Brucella vaccination status (Yes No)').classes_)).index(st.session_state['form_data']['vaccine']) if le_dict and st.session_state['form_data']['vaccine'] else 0, key="vaccine_input")
                sample = st.selectbox(t["sample"], options=sorted(list(le_dict.get('Sample Type(Serum Milk)').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Sample Type(Serum Milk)').classes_)).index(st.session_state['form_data']['sample']) if le_dict and st.session_state['form_data']['sample'] else 0, key="sample_input")
                test = st.selectbox(t["test"], options=sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)').classes_)).index(st.session_state['form_data']['test']) if le_dict and st.session_state['form_data']['test'] else 0, key="test_input")
                retained = st.selectbox(t["retained"], options=sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)').classes_)).index(st.session_state['form_data']['retained']) if le_dict and st.session_state['form_data']['retained'] else 0, key="retained_input")
                disposal = st.selectbox(t["disposal"], options=sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)').classes_)) if le_dict else ["Loading..."], index=sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)').classes_)).index(st.session_state['form_data']['disposal']) if le_dict and st.session_state['form_data']['disposal'] else 0, key="disposal_input")
            
            # Update session state with current form values
            st.session_state['form_data'].update({
                'age': age,
                'breed': breed,
                'sex': sex,
                'calvings': calvings,
                'abortion': abortion,
                'infertility': infertility,
                'vaccine': vaccine,
                'sample': sample,
                'test': test,
                'retained': retained,
                'disposal': disposal
            })
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        run_prediction_btn = st.button(f"🔬 {t['run_prediction']}", use_container_width=True, type="primary", key="run_pred_btn")
        
        if run_prediction_btn:
            input_data = {
                'Age': age, 'Breed species': breed, 'Sex': sex, 'Calvings': calvings,
                'Abortion History (Yes No)': abortion, 'Infertility Repeat breeder(Yes No)': infertility,
                'Brucella vaccination status (Yes No)': vaccine, 'Sample Type(Serum Milk)': sample,
                'Test Type (RBPT ELISA MRT)': test, 'Retained Placenta Stillbirth(Yes No No Data)': retained,
                'Proper Disposal of Aborted Fetuses (Yes No)': disposal
            }
            
            input_df = pd.DataFrame([input_data])
            
            for col in input_df.columns:
                if col in le_dict and input_df[col].dtype == 'object':
                    input_df[col] = le_dict[col].transform(input_df[col])
            
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            
            is_linear = isinstance(best_model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))
            processed = scaler.transform(input_df) if is_linear else input_df.values
            
            try:
                pred_idx = best_model.predict(processed)[0]
                probs = best_model.predict_proba(processed)[0]
                res_label = le_target.inverse_transform([pred_idx])[0]
                conf_score = probs.max()
                
                st.session_state['last_prediction'] = {
                    'result': res_label,
                    'confidence': conf_score,
                    'probabilities': probs,
                    'classes': le_target.classes_,
                    'input_data': input_data
                }
                st.session_state['prediction_count'] += 1
                if "Positive" in res_label:
                    st.session_state['positive_count'] += 1
                st.rerun()
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
        
        # Results Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="section-card">
            <div class="section-header">📊 {t["prediction_results"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get('last_prediction'):
            pred = st.session_state['last_prediction']
            result_type = "positive" if "Positive" in pred['result'] else "negative"
            
            status_emoji = "🔴" if result_type == "positive" else "✅"
            status_text = "POSITIVE" if result_type == "positive" else "NEGATIVE"
            
            st.markdown(f"""
            <div class="prediction-result {result_type}">
                <div class="result-status">{status_emoji} {status_text}</div>
                <div class="result-confidence">{t['confidence']} {pred['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution Chart
            st.markdown(f"<br><h3 class='section-header'>📈 {t['probability_dist']}</h3>", unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                'Class': pred['classes'],
                'Probability': pred['probabilities'] * 100
            })
            
            fig = go.Figure(data=[
                go.Bar(
                    x=prob_df['Class'],
                    y=prob_df['Probability'],
                    marker=dict(
                        color=prob_df['Probability'],
                        colorscale=[[0, '#10b981'], [0.5, '#fbbf24'], [1, '#ef4444']],
                        line=dict(color='rgba(0,0,0,0.1)', width=2)
                    ),
                    text=[f"{val:.1f}%" for val in prob_df['Probability']],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="",
                yaxis_title="Probability (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", size=12),
                showlegend=False,
                yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights
            if ai_enabled:
                st.markdown(f"<br><h3 class='section-header'>💡 {t['ai_insights']}</h3>", unsafe_allow_html=True)
                with st.spinner("🤖 Generating AI recommendations..."):
                    try:
                        prompt = f"""You are a senior veterinary expert. Analyzing animal data: {json.dumps(pred['input_data'])}. 
                        Prediction Result: {pred['result']}. Confidence: {pred['confidence']*100:.1f}%. 
                        If result is Positive, strongly advise immediate isolation and confirmatory lab testing. 
                        Provide 3-4 clear, actionable steps for the farmer in {'Hindi' if selected_lang == 'Hindi' else 'English'}."""
                        
                        response = gemini_model.generate_content(prompt)
                        st.markdown(f'<div class="info-card">{response.text}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"AI Generation Error: {e}")
        else:
            st.markdown(f"""
            <div class="section-card">
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <div class="empty-state-text">{t["run_prediction_msg"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        # AI Assistant Card
        if ai_enabled:
            st.markdown(f"""
            <div class="ai-card">
                <div class="ai-card-icon">🤖</div>
                <div class="ai-badge">
                    ✨ AI Powered
                </div>
                <div class="ai-card-title">{t["vet_assistant"]}</div>
                <div class="ai-card-subtitle">{t["vet_subtitle"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"💬 {t['start_consultation']}", use_container_width=True, key="toggle_chat_btn"):
                st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
                st.rerun()
            
            if st.session_state['show_chatbot']:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                if len(st.session_state['chat_history']) == 0:
                    st.markdown("""
                    <div class="empty-state">
                        <div class="empty-state-icon">💬</div>
                        <div class="empty-state-text">Start a conversation</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for msg in st.session_state['chat_history']:
                        msg_class = "user" if msg['role'] == 'user' else "assistant"
                        role_label = "You" if msg['role'] == 'user' else "AI Assistant"
                        st.markdown(f'<div class="chat-message {msg_class}"><strong>{role_label}</strong>{msg["content"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                user_question = st.text_input("💬 Your question...", key="chat_input")
                
                col_send, col_clear = st.columns([3, 1])
                with col_send:
                    if st.button("Send", use_container_width=True) and user_question:
                        st.session_state['chat_history'].append({"role": "user", "content": user_question})
                        
                        with st.spinner("Thinking..."):
                            try:
                                system_prompt = f"""You are a veterinary consultant specializing in Brucellosis and dairy animal health. 
                                Answer questions about: Brucellosis disease, symptoms, transmission, prevention, vaccination, milk safety, 
                                treatment, diagnosis tests, farm biosecurity, and cattle/buffalo health. 
                                Provide clear, practical advice in {'Hindi' if selected_lang == 'Hindi' else 'English'}. 
                                Keep answers concise (3-5 sentences)."""
                                
                                full_prompt = f"{system_prompt}\n\nUser Question: {user_question}"
                                response = gemini_model.generate_content(full_prompt)
                                st.session_state['chat_history'].append({"role": "assistant", "content": response.text})
                                st.session_state['ai_consultation_count'] += 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"Chat Error: {e}")
                
                with col_clear:
                    if st.button("Clear", use_container_width=True):
                        st.session_state['chat_history'] = []
                        st.rerun()
        
        # Quick Actions
        st.markdown(f"<br><h3 class='section-header'>⚡ {t['quick_actions']}</h3>", unsafe_allow_html=True)
        
        quick_actions = [
            ("📄", t["export_report"]),
            ("📅", t["schedule_test"]),
            ("📋", t["view_guidelines"])
        ]
        
        for icon, label in quick_actions:
            st.markdown(f"""
            <div class="quick-action">
                <div class="quick-action-icon">{icon}</div>
                <div class="quick-action-label">{label}</div>
                <span style="color: #cbd5e1; font-size: 1.25rem;">›</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #64748b; font-size: 0.9rem;">
        <p style="margin: 0;">Developed for Veterinary Health Solutions</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">© 2024 BrucellosisAI. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
