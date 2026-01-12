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
        "run_prediction": "🔬 Run AI Prediction",
        "prediction_results": "Prediction Results",
        "probability_dist": "Probability Distribution",
        "run_prediction_msg": "Run a prediction to see results",
        "vet_assistant": "Veterinary AI Assistant",
        "vet_subtitle": "Get instant expert advice on brucellosis, milk safety, and animal health",
        "start_consultation": "🩺 Start Consultation",
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
        "resources": "RESOURCES"
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
        "run_prediction": "🔬 AI भविष्यवाणी चलाएं",
        "prediction_results": "भविष्यवाणी परिणाम",
        "probability_dist": "संभावना वितरण",
        "run_prediction_msg": "परिणाम देखने के लिए भविष्यवाणी चलाएं",
        "vet_assistant": "पशु चिकित्सा AI सहायक",
        "vet_subtitle": "ब्रुसेलोसिस, दूध सुरक्षा और पशु स्वास्थ्य पर तुरंत विशेषज्ञ सलाह प्राप्त करें",
        "start_consultation": "🩺 परामर्श शुरू करें",
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
        "resources": "संसाधन"
    }
}

# --- SESSION STATE ---
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

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        padding: 0;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Dashboard Header */
    .dashboard-header {
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .dashboard-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
        margin: 0;
    }
    
    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin: 0.25rem 0 0 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-card.blue { border-left-color: #3b82f6; }
    .metric-card.red { border-left-color: #ef4444; }
    .metric-card.green { border-left-color: #10b981; }
    .metric-card.purple { border-left-color: #8b5cf6; }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-change {
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    
    .metric-icon {
        font-size: 1.5rem;
        opacity: 0.7;
    }
    
    /* Input Section */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    
    /* Results Card */
    .results-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        min-height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .empty-state {
        color: #9ca3af;
        font-size: 1rem;
    }
    
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    /* Prediction Result */
    .prediction-result {
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .prediction-result.positive {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
    }
    
    .prediction-result.negative {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
    }
    
    .result-status {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .result-confidence {
        font-size: 1.25rem;
        opacity: 0.8;
    }
    
    /* AI Assistant Card */
    .ai-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .ai-card-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .ai-card-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 1.5rem;
    }
    
    /* Quick Actions */
    .quick-action {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .quick-action:hover {
        background: #f9fafb;
        border-color: #3b82f6;
        transform: translateX(4px);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sidebar Menu */
    .sidebar-menu-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        color: white;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .sidebar-menu-item:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-menu-item.active {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .sidebar-section-title {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 1rem 1rem 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Chat Interface */
    .chat-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        max-height: 400px;
        overflow-y: auto;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .chat-message.user {
        background: #eff6ff;
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background: #f9fafb;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- LANGUAGE SELECTOR ---
selected_lang = st.sidebar.selectbox("🌐 Language / भाषा", ["English", "Hindi"], label_visibility="collapsed")
t = translations[selected_lang]

# --- LOGIN/REGISTER ---
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 3rem 0;">
            <h1 style="font-size: 3rem; font-weight: 700; color: #1e3c72; margin-bottom: 0.5rem;">
                🔬 BrucellosisAI
            </h1>
            <p style="font-size: 1.1rem; color: #6b7280;">
                Advanced Disease Prediction System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.form("login_form"):
                st.text_input("📧 Email Address", key="login_email")
                st.text_input("🔒 Password", type="password", key="login_password")
                submit = st.form_submit_button("Login", use_container_width=True, type="primary")
                
                if submit:
                    try:
                        with open(USERS_FILE, 'r') as f:
                            users = json.load(f)
                        if st.session_state.login_email in users and pbkdf2_sha256.verify(st.session_state.login_password, users[st.session_state.login_email]):
                            st.session_state.update(logged_in=True, username=st.session_state.login_email)
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password")
                    except:
                        st.error("❌ User database not found")
        
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

else:
    # --- SIDEBAR MENU ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">🔬 BrucellosisAI</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0; font-size: 0.85rem;">Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section-title">MAIN MENU</div>', unsafe_allow_html=True)
        
        menu_items = [
            ("📊", t["dashboard_menu"], "dashboard"),
            ("➕", t["new_prediction"], "new_prediction"),
            ("📜", t["history"], "history"),
            ("📈", t["analytics"], "analytics"),
        ]
        
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = 'dashboard'
        
        for icon, label, key in menu_items:
            active_class = "active" if st.session_state['current_page'] == key else ""
            if st.button(f"{icon} {label}", key=f"menu_{key}", use_container_width=True):
                st.session_state['current_page'] = key
                st.rerun()
        
        st.markdown(f'<div class="sidebar-section-title">{t["resources"]}</div>', unsafe_allow_html=True)
        
        if ai_enabled:
            if st.button(f"🤖 {t['ai_assistant']}", key="menu_ai", use_container_width=True):
                st.session_state['current_page'] = 'ai_assistant'
                st.rerun()
        
        if st.button(f"📋 {t['guidelines']}", key="menu_guidelines", use_container_width=True):
            st.session_state['current_page'] = 'guidelines'
            st.rerun()
        
        if st.button(f"⚙️ {t['settings']}", key="menu_settings", use_container_width=True):
            st.session_state['current_page'] = 'settings'
            st.rerun()
        
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px; margin-top: auto;">
            <p style="color: white; margin: 0; font-size: 0.85rem;">👤 Dr. User</p>
            <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem;">Veterinarian</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"🚪 {t['logout']}", use_container_width=True, type="primary"):
            st.session_state.update(logged_in=False, username=None)
            st.rerun()
    
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
            <div class="metric-icon">📊</div>
            <div class="metric-value">{st.session_state['prediction_count']:,}</div>
            <div class="metric-label">{t["total_predictions"]}</div>
            <div class="metric-change" style="color: #10b981;">↑ 45.2% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card red">
            <div class="metric-icon">⚠️</div>
            <div class="metric-value">{st.session_state['positive_count']}</div>
            <div class="metric-label">{t["positive_cases"]}</div>
            <div class="metric-change" style="color: #ef4444;">↑ 8.2% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-icon">✓</div>
            <div class="metric-value">94.3%</div>
            <div class="metric-label">{t["accuracy_rate"]}</div>
            <div class="metric-change" style="color: #10b981;">↑ 0.8% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="metric-icon">🤖</div>
            <div class="metric-value">{st.session_state['ai_consultation_count']}</div>
            <div class="metric-label">{t["ai_consultations"]}</div>
            <div class="metric-change" style="color: #8b5cf6;">↑ 28.7% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Input Section
        st.markdown(f"""
        <div class="input-section">
            <h2 class="section-header">{t["input_header"]}</h2>
            <p class="section-subtitle">{t["input_subtitle"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            age = st.number_input(t["age"], min_value=0, max_value=20, value=5, step=1)
            breed = st.selectbox(t["breed"], options=sorted(list(le_dict.get('Breed species').classes_)) if le_dict else ["Loading..."])
            sex = st.selectbox(t["sex"], options=sorted(list(le_dict.get('Sex').classes_)) if le_dict else ["Loading..."])
            calvings = st.number_input(t["calvings"], min_value=0, max_value=15, value=1, step=1)
            abortion = st.selectbox(t["abortion"], options=sorted(list(le_dict.get('Abortion History (Yes No)').classes_)) if le_dict else ["Loading..."])
            infertility = st.selectbox(t["infertility"], options=sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)').classes_)) if le_dict else ["Loading..."])
        
        with col_b:
            vaccine = st.selectbox(t["vaccination"], options=sorted(list(le_dict.get('Brucella vaccination status (Yes No)').classes_)) if le_dict else ["Loading..."])
            sample = st.selectbox(t["sample"], options=sorted(list(le_dict.get('Sample Type(Serum Milk)').classes_)) if le_dict else ["Loading..."])
            test = st.selectbox(t["test"], options=sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)').classes_)) if le_dict else ["Loading..."])
            retained = st.selectbox(t["retained"], options=sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)').classes_)) if le_dict else ["Loading..."])
            disposal = st.selectbox(t["disposal"], options=sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)').classes_)) if le_dict else ["Loading..."])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button(t["run_prediction"], use_container_width=True, type="primary"):
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
        <div class="input-section">
            <h2 class="section-header">{t["prediction_results"]}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get('last_prediction'):
            pred = st.session_state['last_prediction']
            result_type = "positive" if "Positive" in pred['result'] else "negative"
            
            st.markdown(f"""
            <div class="prediction-result {result_type}">
                <div class="result-status">
                    {"🔴 POSITIVE" if result_type == "positive" else "✅ NEGATIVE"}
                </div>
                <div class="result-confidence">
                    Confidence: {pred['confidence']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution Chart
            st.markdown(f"<br><h3 class='section-header'>{t['probability_dist']}</h3>", unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                'Class': pred['classes'],
                'Probability': pred['probabilities']
            })
            
            fig = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                color='Probability',
                color_continuous_scale=['#10b981', '#fbbf24', '#ef4444'],
                labels={'Probability': 'Probability', 'Class': ''},
                title='Last 7 days'
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights
            if ai_enabled:
                st.markdown(f"<br><h3 class='section-header'>{t['ai_insights']}</h3>", unsafe_allow_html=True)
                with st.spinner("🤖 Generating AI recommendations..."):
                    try:
                        prompt = f"""You are a senior veterinary expert. Analyzing animal data: {json.dumps(pred['input_data'])}. 
                        Prediction Result: {pred['result']}. Confidence: {pred['confidence']*100:.1f}%. 
                        If result is Positive, strongly advise immediate isolation and confirmatory lab testing. 
                        Provide 3-4 clear, actionable steps for the farmer in {'Hindi' if selected_lang == 'Hindi' else 'English'}."""
                        
                        response = gemini_model.generate_content(prompt)
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"AI Generation Error: {e}")
        else:
            st.markdown(f"""
            <div class="results-card">
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <div>{t["run_prediction_msg"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        # AI Assistant Card
        if ai_enabled:
            st.markdown(f"""
            <div class="ai-card">
                <div class="ai-card-title">🤖 {t["vet_assistant"]}</div>
                <div class="ai-card-subtitle">{t["vet_subtitle"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(t["start_consultation"], use_container_width=True, key="start_consult"):
                st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
            
            if st.session_state['show_chatbot']:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for msg in st.session_state['chat_history']:
                    msg_class = "user" if msg['role'] == 'user' else "assistant"
                    st.markdown(f'<div class="chat-message {msg_class}">{msg["content"]}</div>', unsafe_allow_html=True)
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
        st.markdown(f"<br><h3 class='section-header'>{t['quick_actions']}</h3>", unsafe_allow_html=True)
        
        quick_actions = [
            ("📄", t["export_report"]),
            ("📅", t["schedule_test"]),
            ("📋", t["view_guidelines"])
        ]
        
        for icon, label in quick_actions:
            st.markdown(f"""
            <div class="quick-action">
                <span style="font-size: 1.5rem;">{icon}</span>
                <span style="flex: 1;">{label}</span>
                <span style="color: #9ca3af;">›</span>
            </div>
            """, unsafe_allow_html=True)
