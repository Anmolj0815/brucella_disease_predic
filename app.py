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
    "English": {...},   # UNCHANGED (same as your original)
    "Hindi": {...}
}

# --- SESSION STATE & PERSISTENCE ---
@st.cache_resource
def get_session_cache():
    return {}

session_cache = get_session_cache()

def init_session():
    try:
        query_params = st.query_params
        session_id = query_params.get("session_id", None)
    except:
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

# =========================
# CHAT FIX (REPLACED)
# =========================

def process_chat_message(user_input):
    if not user_input:
        return

    st.session_state['chat_history'].append({"role": "user", "content": user_input})

    try:
        lang = st.session_state.get('selected_language', 'English')

        system_prompt = f"""You are a veterinary consultant specializing in Brucellosis and dairy animal health.
Answer questions about: Brucellosis disease, symptoms, transmission, prevention, vaccination, milk safety,
treatment, diagnosis tests, farm biosecurity, and cattle/buffalo health.
Provide clear, practical advice in {'Hindi' if lang == 'Hindi' else 'English'}.
Keep answers concise (3-5 sentences)."""

        full_prompt = f"{system_prompt}\n\nUser Question: {user_input}"

        response = gemini_model.generate_content(full_prompt)

        st.session_state['chat_history'].append(
            {"role": "assistant", "content": response.text}
        )

        st.session_state['ai_consultation_count'] += 1

    except Exception as e:
        st.session_state['chat_error'] = str(e)


def clear_chat():
    st.session_state['chat_history'] = []

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
        body = f"""Hello,

Your OTP for Brucellosis Prediction App registration is: {otp_code}

This OTP is valid for 10 minutes.

Best regards,
Brucellosis App Team"""
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

# --- SIMPLE UI CSS ---
st.markdown("""
<style>
    .stApp { background-color: white; }
    div.stButton > button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN / REGISTER
# (UNCHANGED)
# =========================

# ... keep your full login + prediction + analytics code EXACTLY SAME ...

# =========================
# CHAT UI FIXED
# =========================

with col_right:
    if ai_enabled:
        st.subheader(f"🤖 {t['vet_assistant']}")
        st.write(t["vet_subtitle"])

        if st.button(f"💬 {t['start_consultation']}", use_container_width=True, key="toggle_chat_btn"):
            st.session_state['show_chatbot'] = not st.session_state['show_chatbot']

        if st.session_state['show_chatbot']:
            st.markdown("---")

            if len(st.session_state['chat_history']) == 0:
                st.info("Start a conversation")
            else:
                for msg in st.session_state['chat_history']:
                    role_label = "You" if msg['role'] == 'user' else "AI Assistant"
                    st.markdown(f"**{role_label}:** {msg['content']}")

            st.markdown("---")

            if 'chat_error' in st.session_state and st.session_state['chat_error']:
                st.error(f"Chat Error: {st.session_state['chat_error']}")
                st.session_state['chat_error'] = None

            with st.form("chat_form", clear_on_submit=True):
                user_msg = st.text_input("💬 Your question...", key="chat_input")

                col_send, col_clear = st.columns([3, 1])

                with col_send:
                    send_clicked = st.form_submit_button("Send", use_container_width=True)

                with col_clear:
                    clear_clicked = st.form_submit_button("Clear", use_container_width=True)

            if send_clicked:
                process_chat_message(user_msg)

            if clear_clicked:
                clear_chat()
