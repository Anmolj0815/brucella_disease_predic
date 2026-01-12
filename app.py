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
import gspread
from google.oauth2.service_account import Credentials
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import time

warnings.filterwarnings('ignore')

# --- GEMINI CONFIGURATION ---
ai_enabled = False
gemini_model = None

if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        try:
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            if available_models:
                model_to_use = available_models[0]
                gemini_model = genai.GenerativeModel(model_name=model_to_use)
                ai_enabled = True
            else:
                st.sidebar.warning("No AI models found with generateContent support.")
                
        except Exception as list_error:
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            for model_name in model_names:
                try:
                    gemini_model = genai.GenerativeModel(model_name=model_name)
                    ai_enabled = True
                    break
                except:
                    continue
            
            if not ai_enabled:
                st.sidebar.warning("Could not initialize AI service. Advanced features disabled.")
            
    except Exception as e:
        st.sidebar.error(f"AI Setup Error: {e}")
else:
    st.sidebar.warning("API Key not found. AI features disabled.")

# --- TRANSLATIONS ---
translations = {
    "English": {
        "welcome": "Welcome to Brucellosis Prediction System",
        "title": "Brucellosis Prediction Model",
        "user_greet": "Welcome back, {}",
        "input_header": "Animal Information",
        "age": "Age (Years)", "breed": "Breed/Species", "sex": "Sex",
        "calvings": "Number of Calvings", "abortion": "Abortion History",
        "infertility": "Infertility/Repeat Breeder", "vaccination": "Vaccination Status",
        "sample": "Sample Type", "test": "Test Type",
        "retained": "Retained Placenta/Stillbirth", "disposal": "Proper Disposal of Aborted Fetuses",
        "predict_btn": "Run Prediction",
        "results_header": "Prediction Results", "pred_res": "Predicted Status:",
        "conf": "Confidence Score:", "prob_header": "Probability Distribution",
        "chart_title": "Class Distribution", "logout": "Logout", "login_sub": "Login",
        "ai_advice_header": "Veterinary Consultation",
        "ai_loading": "Analyzing data and generating recommendations...",
        "system_prompt": "You are a senior veterinary expert. Analyzing animal data: {}. Prediction Result: {}. Confidence: {}%. If result is Positive, strongly advise immediate isolation and confirmatory lab testing (RBPT/ELISA). Provide 3-4 clear, actionable steps for the farmer in English.",
        "chatbot_button": "Ask Veterinary Assistant",
        "chatbot_title": "Veterinary Assistant",
        "chatbot_subtitle": "Ask questions about Brucellosis, milk safety, and animal health",
        "chat_placeholder": "Type your question here...",
        "chat_system": "You are a veterinary consultant specializing in Brucellosis and dairy animal health. Answer questions about: Brucellosis disease, symptoms in animals, transmission, prevention, vaccination, milk safety, treatment, diagnosis tests (RBPT/ELISA/MRT), farm biosecurity, and general cattle/buffalo health. Provide clear, practical advice in English. Keep answers concise (3-5 sentences) unless detailed explanation is requested."
    },
    "Hindi": {
        "welcome": "ब्रुसेलोसिस भविष्यवाणी प्रणाली में आपका स्वागत है",
        "title": "ब्रुसेलोसिस भविष्यवाणी मॉडल",
        "user_greet": "आपका स्वागत है, {}",
        "input_header": "पशु जानकारी",
        "age": "आयु (वर्ष)", "breed": "नस्ल/प्रजाति", "sex": "लिंग",
        "calvings": "बछड़े की संख्या", "abortion": "गर्भपात का इतिहास",
        "infertility": "बांझपन", "vaccination": "टीकाकरण की स्थिति",
        "sample": "नमूना प्रकार", "test": "परीक्षण प्रकार",
        "retained": "जेर रुकना/मृत प्रसव", "disposal": "भ्रूण का निपटान",
        "predict_btn": "भविष्यवाणी करें",
        "results_header": "परिणाम", "pred_res": "अनुमानित स्थिति:",
        "conf": "भरोसा:", "prob_header": "संभावना विश्लेषण",
        "chart_title": "संभावना चार्ट", "logout": "लॉगआउट", "login_sub": "लॉगिन",
        "ai_advice_header": "पशु चिकित्सक सलाह",
        "ai_loading": "डेटा का विश्लेषण और सुझाव तैयार किए जा रहे हैं...",
        "system_prompt": "आप एक वरिष्ठ पशु चिकित्सा विशेषज्ञ हैं। पशु डेटा: {}. भविष्यवाणी परिणाम: {}. भरोसा: {}%. यदि परिणाम पॉजिटिव है, तो तुरंत पशु को अलग करने और लैब टेस्टिंग की सलाह दें। किसान के लिए हिंदी में 3-4 स्पष्ट और व्यावहारिक सुझाव दें।",
        "chatbot_button": "पशु चिकित्सक से पूछें",
        "chatbot_title": "पशु चिकित्सा सहायक",
        "chatbot_subtitle": "ब्रुसेलोसिस, दूध की सुरक्षा और पशु स्वास्थ्य के बारे में पूछें",
        "chat_placeholder": "अपना प्रश्न यहाँ लिखें...",
        "chat_system": "आप एक पशु चिकित्सा सलाहकार हैं जो ब्रुसेलोसिस और डेयरी पशु स्वास्थ्य में विशेषज्ञता रखते हैं। इन विषयों पर सवालों के जवाब दें: ब्रुसेलोसिस रोग, पशुओं में लक्षण, संचरण, रोकथाम, टीकाकरण, दूध की सुरक्षा, उपचार, निदान परीक्षण, फार्म बायोसिक्योरिटी, और सामान्य गाय/भैंस स्वास्थ्य। हिंदी में स्पष्ट, व्यावहारिक सलाह दें।"
    }
}

# --- MODEL LOADING ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'
GOOGLE_SHEET_ID = '159z65oDmaBPymwndIHkNVbK1Q6_GMmFc7xGcJ2fsozY'

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'show_chatbot' not in st.session_state:
    st.session_state['show_chatbot'] = False
if 'otp_sent' not in st.session_state:
    st.session_state['otp_sent'] = False
if 'otp_code' not in st.session_state:
    st.session_state['otp_code'] = None
if 'otp_timestamp' not in st.session_state:
    st.session_state['otp_timestamp'] = None
if 'pending_user_data' not in st.session_state:
    st.session_state['pending_user_data'] = None

def generate_otp():
    """Generate 6-digit OTP"""
    return str(random.randint(100000, 999999))

def send_otp_email(recipient_email, otp_code):
    """Send OTP via email"""
    try:
        smtp_user = st.secrets["email"]["smtp_user"]
        smtp_password = st.secrets["email"]["smtp_password"]
        
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = recipient_email
        msg['Subject'] = "Brucellosis App - Email Verification OTP"
        
        body = f"""
        Hello,
        
        Your OTP for Brucellosis Prediction App registration is: {otp_code}
        
        This OTP is valid for 10 minutes.
        
        If you did not request this, please ignore this email.
        
        Best regards,
        Brucellosis App Team
        """
        
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
    """Verify if entered OTP is correct and not expired"""
    if st.session_state['otp_code'] is None:
        return False, "No OTP sent"
    
    if time.time() - st.session_state['otp_timestamp'] > 600:
        return False, "OTP expired. Please request a new one."
    
    if entered_otp == st.session_state['otp_code']:
        return True, "OTP verified successfully"
    else:
        return False, "Invalid OTP. Please try again."

def connect_to_google_sheet():
    """Connect to Google Sheets"""
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1
        return sheet
    except Exception as e:
        st.error(f"Google Sheets connection error: {e}")
        return None

def save_user_to_google_sheet(email, name, phone, location):
    """Save new user registration to Google Sheet"""
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
        
        st.success("User data saved successfully")
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheet: {e}")
        return False

def register_user(email, password, name, phone, location):
    """Register new user in JSON and Google Sheet"""
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
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .positive-result {
        background-color: #fee2e2;
        border-left-color: #dc2626;
    }
    .negative-result {
        background-color: #d1fae5;
        border-left-color: #059669;
    }
    .info-card {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .chat-container {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .chat-message {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
    .user-message {
        background-color: #dbeafe;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f3f4f6;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- UI LOGIC ---
st.set_page_config(page_title="Brucellosis Prediction System", layout="wide", initial_sidebar_state="expanded")
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
t = translations[selected_lang]

if not st.session_state['logged_in']:
    st.markdown(f'<div class="main-header">{t["welcome"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Veterinary diagnostic system for Brucellosis detection</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader(t["login_sub"])
        with st.form("login_form"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                try:
                    with open(USERS_FILE, 'r') as f: users = json.load(f)
                    if email in users and pbkdf2_sha256.verify(password, users[email]):
                        st.session_state.update(logged_in=True, username=email)
                        st.rerun()
                    else: 
                        st.error("Invalid email or password")
                except: 
                    st.error("User database not found")
    
    with tab2:
        st.subheader("Create New Account")
        
        if not st.session_state['otp_sent']:
            with st.form("register_form"):
                reg_name = st.text_input("Full Name")
                reg_email = st.text_input("Email Address")
                reg_phone = st.text_input("Phone Number")
                reg_location = st.text_input("Location (City/Village)")
                reg_password = st.text_input("Password", type="password")
                reg_confirm = st.text_input("Confirm Password", type="password")
                submit_reg = st.form_submit_button("Send Verification Code", use_container_width=True)
                
                if submit_reg:
                    if not all([reg_name, reg_email, reg_phone, reg_location, reg_password]):
                        st.error("Please fill in all fields")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        try:
                            if os.path.exists(USERS_FILE):
                                with open(USERS_FILE, 'r') as f:
                                    users = json.load(f)
                                if reg_email in users:
                                    st.error("This email is already registered")
                                else:
                                    otp = generate_otp()
                                    if send_otp_email(reg_email, otp):
                                        st.session_state['otp_code'] = otp
                                        st.session_state['otp_timestamp'] = time.time()
                                        st.session_state['otp_sent'] = True
                                        st.session_state['pending_user_data'] = {
                                            'email': reg_email,
                                            'password': reg_password,
                                            'name': reg_name,
                                            'phone': reg_phone,
                                            'location': reg_location
                                        }
                                        st.success(f"Verification code sent to {reg_email}")
                                        st.rerun()
                            else:
                                otp = generate_otp()
                                if send_otp_email(reg_email, otp):
                                    st.session_state['otp_code'] = otp
                                    st.session_state['otp_timestamp'] = time.time()
                                    st.session_state['otp_sent'] = True
                                    st.session_state['pending_user_data'] = {
                                        'email': reg_email,
                                        'password': reg_password,
                                        'name': reg_name,
                                        'phone': reg_phone,
                                        'location': reg_location
                                    }
                                    st.success(f"Verification code sent to {reg_email}")
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        else:
            st.info(f"Verification code sent to {st.session_state['pending_user_data']['email']}")
            st.caption("Enter the 6-digit code sent to your email (valid for 10 minutes)")
            
            entered_otp = st.text_input("Verification Code", max_chars=6)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Verify Code", use_container_width=True):
                    if entered_otp:
                        is_valid, message = verify_otp(entered_otp)
                        if is_valid:
                            user_data = st.session_state['pending_user_data']
                            success, reg_message = register_user(
                                user_data['email'],
                                user_data['password'],
                                user_data['name'],
                                user_data['phone'],
                                user_data['location']
                            )
                            if success:
                                st.success(reg_message + " Please login now.")
                                st.session_state['otp_sent'] = False
                                st.session_state['otp_code'] = None
                                st.session_state['otp_timestamp'] = None
                                st.session_state['pending_user_data'] = None
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(reg_message)
                        else:
                            st.error(message)
                    else:
                        st.error("Please enter the verification code")
            
            with col2:
                if st.button("Resend Code", use_container_width=True):
                    otp = generate_otp()
                    if send_otp_email(st.session_state['pending_user_data']['email'], otp):
                        st.session_state['otp_code'] = otp
                        st.session_state['otp_timestamp'] = time.time()
                        st.success("New code sent")
                        st.rerun()
            
            if st.button("Back to Registration", use_container_width=True):
                st.session_state['otp_sent'] = False
                st.session_state['otp_code'] = None
                st.session_state['otp_timestamp'] = None
                st.session_state['pending_user_data'] = None
                st.rerun()
else:
    st.markdown(f'<div class="main-header">{t["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{t["user_greet"].format(st.session_state["username"])}</div>', unsafe_allow_html=True)
    
    if st.sidebar.button(t["logout"], use_container_width=True):
        st.session_state.update(logged_in=False, username=None)
        st.rerun()

    st.sidebar.header(t["input_header"])
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(t["age"], min_value=0, max_value=20, value=5, step=1)
        breed = st.selectbox(t["breed"], options=sorted(list(le_dict.get('Breed species').classes_)))
        sex = st.selectbox(t["sex"], options=sorted(list(le_dict.get('Sex').classes_)))
        calvings = st.number_input(t["calvings"], min_value=0, max_value=15, value=1, step=1)
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

    if st.button(t["predict_btn"], use_container_width=True, type="primary"):
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

            st.markdown("---")
            st.subheader(t["results_header"])
            
            ui_res = "पॉजिटिव (Positive)" if (selected_lang == "Hindi" and "Positive" in res_label) else \
                     "नेगेटिव (Negative)" if (selected_lang == "Hindi" and "Negative" in res_label) else res_label
            
            result_class = "positive-result" if "Positive" in res_label else "negative-result"
            
            st.markdown(f'''
            <div class="result-box {result_class}">
                <h3>{t["pred_res"]} {ui_res}</h3>
                <p>{t["conf"]} {conf_score:.2%}</p>
            </div>
            ''', unsafe_allow_html=True)

            if ai_enabled:
                st.subheader(t["ai_advice_header"])
                with st.spinner(t["ai_loading"]):
                    try:
                        auto_prompt = t["system_prompt"].format(json.dumps(input_data), res_label, round(conf_score*100, 2))
                        response = gemini_model.generate_content(auto_prompt)
                        st.markdown(f'<div class="info-card">{response.text}</div>', unsafe_allow_html=True)
                    except Exception as ai_e:
                        st.error(f"AI Generation Error: {ai_e}")

            st.write("---")
            st.subheader(t["prob_header"])
            prob_df = pd.DataFrame({'Probability': probs}, index=le_target.classes_)
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(x=prob_df.index, y=prob_df['Probability'], palette='viridis', ax=ax)
            ax.set_ylabel('Probability')
            ax.set_xlabel('Class')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    if ai_enabled:
        st.markdown("---")
        if st.button(t["chatbot_button"], use_container_width=True):
            st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
        
        if st.session_state['show_chatbot']:
            st.subheader(t["chatbot_title"])
            st.caption(t["chatbot_subtitle"])
            
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for msg in st.session_state['chat_history']:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            user_question = st.text_input(t["chat_placeholder"])
            
            col_send, col_clear = st.columns([4, 1])
            with col_send:
                if st.button("Send", use_container_width=True) and user_question:
                    st.session_state['chat_history'].append({"role": "user", "content": user_question})
                    
                    with st.spinner("Generating response..."):
                        try:
                            full_prompt = f"{t['chat_system']}\n\nUser Question: {user_question}"
                            response = gemini_model.generate_content(full_prompt)
                            ai_response = response.text
                            st.session_state['chat_history'].append({"role": "assistant", "content": ai_response})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Chat Error: {e}")
            
            with col_clear:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state['chat_history'] = []
                    st.rerun()
    
    st.markdown("---")
    st.markdown("Developed for Veterinary Health Solutions")
