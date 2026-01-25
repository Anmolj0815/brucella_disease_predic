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

    # Initialize all keys
    defaults = {
        'logged_in': False,
        'chat_history': [],
        'show_chatbot': False,
        'prediction_count': 1247,
        'positive_count': 87,
        'ai_consultation_count': 342,
        'last_prediction': None,
        'otp_sent': False,
        'otp_code': None,
        'otp_timestamp': None,
        'pending_user_data': None,
        'current_page': 'dashboard',
        'form_data': {
            'age': 5, 'breed': None, 'sex': None, 'calvings': 1,
            'abortion': None, 'infertility': None, 'vaccine': None,
            'sample': None, 'test': None, 'retained': None, 'disposal': None
        }
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

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

# --- CHAT CALLBACKS ---
def handle_chat_submit():
    user_input = st.session_state.chat_input
    if user_input:
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        try:
            lang = st.session_state.get('selected_language', 'English')
            system_prompt = f"""You are a veterinary consultant specializing in Brucellosis. 
            Provide concise practical advice in {'Hindi' if lang == 'Hindi' else 'English'}."""
            full_prompt = f"{system_prompt}\n\nUser Question: {user_input}"
            response = gemini_model.generate_content(full_prompt)
            st.session_state['chat_history'].append({"role": "assistant", "content": response.text})
            st.session_state['ai_consultation_count'] += 1
        except Exception as e:
            st.session_state['chat_error'] = str(e)

def handle_chat_clear():
    st.session_state['chat_history'] = []

# --- DATABASE & AUTH FUNCTIONS ---
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
        msg['Subject'] = "Brucellosis App - OTP"
        body = f"Your OTP is: {otp_code}"
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

def verify_otp(entered_otp):
    if st.session_state['otp_code'] is None:
        return False, "No OTP sent"
    if time.time() - st.session_state['otp_timestamp'] > 600:
        return False, "OTP expired"
    if entered_otp == st.session_state['otp_code']:
        return True, "Verified"
    return False, "Invalid OTP"

def connect_to_google_sheet():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        return client.open_by_key(GOOGLE_SHEET_ID).sheet1
    except:
        return None

def save_user_to_google_sheet(email, name, phone, location):
    sheet = connect_to_google_sheet()
    if sheet:
        reg_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([email, name, phone, location, reg_date])
        return True
    return False

def register_user(email, password, name, phone, location):
    try:
        users = {}
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
        if email in users:
            return False, "User exists"
        users[email] = pbkdf2_sha256.hash(password)
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
        save_user_to_google_sheet(email, name, phone, location)
        return True, "Success"
    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_all_artifacts():
    try:
        with open(MODEL_ARTIFACTS_DIR + 'best_model.pkl', 'rb') as f: m = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'le_dict.pkl', 'rb') as f: ld = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'le_target.pkl', 'rb') as f: lt = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'scaler.pkl', 'rb') as f: s = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'feature_names.pkl', 'rb') as f: fn = pickle.load(f)
        return m, ld, lt, s, fn
    except:
        return None, None, None, None, None

best_model, le_dict, le_target, scaler, feature_names = load_all_artifacts()

# --- UI STYLING ---
st.markdown("""
<style>
    .stApp { background-color: white; }
    div.stButton > button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- LOGIN / SIGNUP LOGIC ---
if not st.session_state['logged_in']:
    col_l1, col_l2, col_l3 = st.columns([2, 1, 2])
    with col_l2:
        selected_lang = st.selectbox("🌐", ["English", "Hindi"], label_visibility="collapsed")
    
    t = translations[selected_lang]
    col1, col2, col3 = st.columns([1, 2.5, 1])
    
    with col2:
        st.title("BrucellosisAI")
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                l_email = st.text_input("Email")
                l_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Access Dashboard", type="primary"):
                    try:
                        with open(USERS_FILE, 'r') as f:
                            users = json.load(f)
                        if l_email in users and pbkdf2_sha256.verify(l_pass, users[l_email]):
                            st.session_state.update(logged_in=True, username=l_email)
                            create_user_session(l_email)
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    except:
                        st.error("Database error")
        
        with tab2:
            if not st.session_state['otp_sent']:
                with st.form("reg_form"):
                    r_name = st.text_input("Full Name")
                    r_email = st.text_input("Email")
                    r_phone = st.text_input("Phone")
                    r_loc = st.text_input("Location")
                    r_pass = st.text_input("Password", type="password")
                    if st.form_submit_button("Send OTP", type="primary"):
                        otp = generate_otp()
                        if send_otp_email(r_email, otp):
                            st.session_state.update(otp_code=otp, otp_sent=True, otp_timestamp=time.time(),
                                                 pending_user_data={'email': r_email, 'password': r_pass, 
                                                                  'name': r_name, 'phone': r_phone, 'location': r_loc})
                            st.rerun()
            else:
                ent_otp = st.text_input("Enter OTP")
                if st.button("Verify & Register"):
                    val, msg = verify_otp(ent_otp)
                    if val:
                        ud = st.session_state['pending_user_data']
                        register_user(ud['email'], ud['password'], ud['name'], ud['phone'], ud['location'])
                        st.success("Registered! Please Login.")
                        st.session_state['otp_sent'] = False
                        time.sleep(1)
                        st.rerun()

# --- MAIN APP ---
else:
    selected_lang = st.sidebar.selectbox("🌐 Language", ["English", "Hindi"], key="selected_language")
    t = translations[selected_lang]
    
    # Sidebar
    with st.sidebar:
        st.title("BrucellosisAI")
        if st.button(f"📊 {t['dashboard_menu']}"): st.session_state['current_page'] = 'dashboard'
        if st.button(f"➕ {t['new_prediction']}"): st.session_state['current_page'] = 'new_prediction'
        if st.button(f"🚪 {t['logout']}", type="primary"): logout_user()

    # Dashboard
    st.title(t["dashboard"])
    st.caption(t["subtitle"])
    
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.subheader(f"📝 {t['input_header']}")
        with st.form("prediction_form"):
            ca, cb = st.columns(2)
            # Input fields (simplified logic for brevity)
            with ca:
                age = st.number_input(t["age"], 0, 20, 5)
                breed = st.selectbox(t["breed"], sorted(list(le_dict.get('Breed species').classes_)) if le_dict else ["..."])
                sex = st.selectbox(t["sex"], sorted(list(le_dict.get('Sex').classes_)) if le_dict else ["..."])
                calvings = st.number_input(t["calvings"], 0, 15, 1)
                abortion = st.selectbox(t["abortion"], sorted(list(le_dict.get('Abortion History (Yes No)').classes_)) if le_dict else ["..."])
                infertility = st.selectbox(t["infertility"], sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)').classes_)) if le_dict else ["..."])
            with cb:
                vaccine = st.selectbox(t["vaccination"], sorted(list(le_dict.get('Brucella vaccination status (Yes No)').classes_)) if le_dict else ["..."])
                sample = st.selectbox(t["sample"], sorted(list(le_dict.get('Sample Type(Serum Milk)').classes_)) if le_dict else ["..."])
                test = st.selectbox(t["test"], sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)').classes_)) if le_dict else ["..."])
                retained = st.selectbox(t["retained"], sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)').classes_)) if le_dict else ["..."])
                disposal = st.selectbox(t["disposal"], sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)').classes_)) if le_dict else ["..."])
            
            if st.form_submit_button(f"🔬 {t['run_prediction']}", type="primary"):
                # Prediction Logic
                input_data = {'Age': age, 'Breed species': breed, 'Sex': sex, 'Calvings': calvings,
                             'Abortion History (Yes No)': abortion, 'Infertility Repeat breeder(Yes No)': infertility,
                             'Brucella vaccination status (Yes No)': vaccine, 'Sample Type(Serum Milk)': sample,
                             'Test Type (RBPT ELISA MRT)': test, 'Retained Placenta Stillbirth(Yes No No Data)': retained,
                             'Proper Disposal of Aborted Fetuses (Yes No)': disposal}
                
                df = pd.DataFrame([input_data])
                for col in df.columns:
                    if col in le_dict and df[col].dtype == 'object':
                        df[col] = le_dict[col].transform(df[col])
                
                df = df.reindex(columns=feature_names, fill_value=0)
                is_lin = isinstance(best_model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))
                proc = scaler.transform(df) if is_lin else df.values
                
                res_idx = best_model.predict(proc)[0]
                probs = best_model.predict_proba(proc)[0]
                st.session_state['last_prediction'] = {
                    'result': le_target.inverse_transform([res_idx])[0],
                    'confidence': probs.max(),
                    'probabilities': probs,
                    'classes': le_target.classes_,
                    'input_data': input_data
                }
                st.rerun()

        # Display Results
        if st.session_state['last_prediction']:
            pred = st.session_state['last_prediction']
            if "Positive" in pred['result']:
                st.error(f"🔴 POSITIVE | Confidence: {pred['confidence']:.1%}")
            else:
                st.success(f"✅ NEGATIVE | Confidence: {pred['confidence']:.1%}")
            
            # Chart
            fig = px.bar(x=pred['classes'], y=pred['probabilities']*100, labels={'x':'Result', 'y':'%'})
            st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader(f"🤖 {t['vet_assistant']}")
        if st.button(f"💬 {t['start_consultation']}"):
            st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
            st.rerun()
            
        if st.session_state['show_chatbot']:
            for msg in st.session_state['chat_history']:
                st.write(f"**{msg['role']}:** {msg['content']}")
            with st.form("chat_box", clear_on_submit=True):
                st.text_input("Ask anything...", key="chat_input")
                st.form_submit_button("Send", on_click=handle_chat_submit)

    # Footer
    st.markdown("<div style='text-align: center; margin-top: 50px;'>© 2024 BrucellosisAI</div>", unsafe_allow_html=True)
