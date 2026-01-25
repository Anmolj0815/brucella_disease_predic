import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import json
import os
import plotly.graph_objects as go
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

# --- CONSTANTS ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'
GOOGLE_SHEET_ID = '159z65oDmaBPymwndIHkNVbK1Q6_GMmFc7xGcJ2fsozY'

# --- GEMINI CONFIGURATION (Lazy Loading) ---
@st.cache_resource
def get_gemini_model():
    try:
        import google.generativeai as genai
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    return genai.GenerativeModel(model_name=m.name)
    except:
        pass
    return None

# --- MODEL LOADING ---
@st.cache_resource
def load_all_artifacts():
    try:
        with open(MODEL_ARTIFACTS_DIR + 'best_model.pkl', 'rb') as f:
            m = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'le_dict.pkl', 'rb') as f:
            ld = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'le_target.pkl', 'rb') as f:
            lt = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'scaler.pkl', 'rb') as f:
            s = pickle.load(f)
        with open(MODEL_ARTIFACTS_DIR + 'feature_names.pkl', 'rb') as f:
            fn = pickle.load(f)
        return m, ld, lt, s, fn
    except Exception as e:
        return None, None, None, None, None

# --- TRANSLATIONS ---
translations = {
    "English": {
        "dashboard": "Brucellosis Prediction Dashboard",
        "subtitle": "AI-powered disease prediction and veterinary consultation",
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
        "vet_subtitle": "Get expert advice on brucellosis and animal health",
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
        "confidence": "Confidence Score:",
        "generate_insights": "Generate AI Insights"
    },
    "Hindi": {
        "dashboard": "ब्रुसेलोसिस भविष्यवाणी डैशबोर्ड",
        "subtitle": "AI-संचालित रोग भविष्यवाणी और पशु चिकित्सा परामर्श",
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
        "vet_subtitle": "ब्रुसेलोसिस और पशु स्वास्थ्य पर विशेषज्ञ सलाह",
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
        "confidence": "भरोसा:",
        "generate_insights": "AI सलाह प्राप्त करें"
    }
}

# --- SESSION STATE ---
@st.cache_resource
def get_session_cache():
    return {}

session_cache = get_session_cache()

def init_session():
    defaults = {
        'logged_in': False,
        'username': None,
        'chat_history': [],
        'show_chatbot': False,
        'last_prediction': None,
        'ai_insight': None,
        'otp_sent': False,
        'otp_code': None,
        'otp_timestamp': None,
        'pending_user_data': None,
        'selected_language': 'English'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Check session persistence
    try:
        session_id = st.query_params.get("session_id", None)
    except:
        try:
            session_id = st.experimental_get_query_params().get("session_id", [None])[0]
        except:
            session_id = None
    
    if session_id and session_id in session_cache:
        if not st.session_state['logged_in']:
            st.session_state['logged_in'] = True
            st.session_state['username'] = session_cache[session_id]['username']

init_session()

# --- HELPER FUNCTIONS ---
def create_user_session(username):
    new_session_id = str(uuid.uuid4())
    session_cache[new_session_id] = {'username': username}
    try:
        st.query_params["session_id"] = new_session_id
    except:
        st.experimental_set_query_params(session_id=new_session_id)

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
        body = f"Hello,\n\nYour OTP is: {otp_code}\n\nValid for 10 minutes.\n\nBrucellosis App Team"
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        return True
    except:
        return False

def verify_otp(entered_otp):
    if not st.session_state['otp_code']:
        return False, "No OTP sent"
    if time.time() - st.session_state['otp_timestamp'] > 600:
        return False, "OTP expired"
    if entered_otp == st.session_state['otp_code']:
        return True, "Verified"
    return False, "Invalid OTP"

def connect_to_google_sheet():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(creds)
        return client.open_by_key(GOOGLE_SHEET_ID).sheet1
    except:
        return None

def register_user(email, password, name, phone, location):
    try:
        users = {}
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
        if email in users:
            return False, "User already exists"
        users[email] = pbkdf2_sha256.hash(password)
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
        # Save to Google Sheet
        sheet = connect_to_google_sheet()
        if sheet:
            sheet.append_row([email, name, phone, location, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')])
        return True, "Registration successful"
    except Exception as e:
        return False, str(e)

# --- CACHED AI GENERATION ---
@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_insight(_model, input_data_str, result, confidence, lang):
    """Generate AI insight - cached to prevent repeated API calls"""
    if _model is None:
        return None
    try:
        prompt = f"""You are a senior veterinary expert. Analyzing animal data: {input_data_str}. 
        Prediction Result: {result}. Confidence: {confidence:.1f}%. 
        If result is Positive, strongly advise immediate isolation and confirmatory lab testing. 
        Provide 3-4 clear, actionable steps for the farmer in {'Hindi' if lang == 'Hindi' else 'English'}."""
        response = _model.generate_content(prompt)
        return response.text
    except:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_chat_response(_model, question, lang):
    """Get chat response - cached"""
    if _model is None:
        return "AI not available"
    try:
        prompt = f"""You are a veterinary consultant specializing in Brucellosis. 
        Answer in {'Hindi' if lang == 'Hindi' else 'English'}. Keep it concise (3-5 sentences).
        Question: {question}"""
        response = _model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- LOAD RESOURCES ---
best_model, le_dict, le_target, scaler, feature_names = load_all_artifacts()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    div.stButton > button { width: 100%; border-radius: 8px; }
    .prediction-positive { padding: 1rem; background: #fee2e2; border-radius: 8px; border-left: 4px solid #ef4444; }
    .prediction-negative { padding: 1rem; background: #dcfce7; border-radius: 8px; border-left: 4px solid #22c55e; }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOGIN PAGE
# ============================================
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔬 BrucellosisAI")
        st.caption("Next Gen Disease Prediction")
        
        lang = st.selectbox("🌐 Language", ["English", "Hindi"], key="login_lang")
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                
                if submitted:
                    try:
                        with open(USERS_FILE, 'r') as f:
                            users = json.load(f)
                        if email in users and pbkdf2_sha256.verify(password, users[email]):
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = email
                            create_user_session(email)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    except:
                        st.error("❌ Login failed")
        
        with tab2:
            if not st.session_state['otp_sent']:
                with st.form("register_form", clear_on_submit=False):
                    reg_name = st.text_input("👤 Full Name")
                    reg_email = st.text_input("📧 Email")
                    reg_phone = st.text_input("📱 Phone")
                    reg_location = st.text_input("📍 Location")
                    reg_password = st.text_input("🔒 Password", type="password")
                    reg_confirm = st.text_input("🔒 Confirm Password", type="password")
                    submitted = st.form_submit_button("Send OTP", use_container_width=True, type="primary")
                    
                    if submitted:
                        if not all([reg_name, reg_email, reg_phone, reg_location, reg_password]):
                            st.error("Fill all fields")
                        elif reg_password != reg_confirm:
                            st.error("Passwords don't match")
                        elif len(reg_password) < 6:
                            st.error("Password too short")
                        else:
                            otp = generate_otp()
                            if send_otp_email(reg_email, otp):
                                st.session_state['otp_code'] = otp
                                st.session_state['otp_timestamp'] = time.time()
                                st.session_state['otp_sent'] = True
                                st.session_state['pending_user_data'] = {
                                    'email': reg_email, 'password': reg_password,
                                    'name': reg_name, 'phone': reg_phone, 'location': reg_location
                                }
                                st.rerun()
                            else:
                                st.error("Failed to send OTP")
            else:
                st.info(f"📧 OTP sent to {st.session_state['pending_user_data']['email']}")
                with st.form("otp_form", clear_on_submit=False):
                    entered_otp = st.text_input("Enter OTP", max_chars=6)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        verify = st.form_submit_button("Verify", use_container_width=True, type="primary")
                    with col_b:
                        resend = st.form_submit_button("Resend", use_container_width=True)
                    
                    if verify and entered_otp:
                        valid, msg = verify_otp(entered_otp)
                        if valid:
                            ud = st.session_state['pending_user_data']
                            success, msg = register_user(ud['email'], ud['password'], ud['name'], ud['phone'], ud['location'])
                            if success:
                                st.success("✅ Registered! Please login.")
                                st.session_state['otp_sent'] = False
                                st.session_state['pending_user_data'] = None
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.error(msg)
                    
                    if resend:
                        otp = generate_otp()
                        if send_otp_email(st.session_state['pending_user_data']['email'], otp):
                            st.session_state['otp_code'] = otp
                            st.session_state['otp_timestamp'] = time.time()
                            st.success("New OTP sent")

# ============================================
# MAIN DASHBOARD
# ============================================
else:
    # Sidebar
    with st.sidebar:
        st.title("🔬 BrucellosisAI")
        
        lang = st.selectbox("🌐", ["English", "Hindi"], key="selected_language", label_visibility="collapsed")
        t = translations[lang]
        
        st.markdown("---")
        
        # Simple navigation with selectbox (no reruns on empty clicks)
        page = st.selectbox(
            "Navigation",
            [f"📊 {t['dashboard_menu']}", f"📜 {t['history']}", f"📈 {t['analytics']}", f"⚙️ {t['settings']}"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.write(f"👤 {st.session_state.get('username', 'User')}")
        
        # Logout with callback for proper handling
        def do_logout():
            # Clear session cache
            try:
                session_id = st.query_params.get("session_id", None)
                if session_id and session_id in session_cache:
                    del session_cache[session_id]
            except:
                pass
            # Clear query params
            try:
                st.query_params.clear()
            except:
                try:
                    st.experimental_set_query_params()
                except:
                    pass
            # Reset login state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state['logged_in'] = False
        
        if st.button(f"🚪 {t['logout']}", use_container_width=True, on_click=do_logout):
            st.rerun()

    
    t = translations[lang]
    
    # Main content
    st.title(t["dashboard"])
    st.caption(t["subtitle"])
    
    # Two columns
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader(f"📝 {t['input_header']}")
        
        if le_dict is None:
            st.error("Model artifacts not found. Please check model_artifacts folder.")
        else:
            # Prediction Form
            with st.form("prediction_form", clear_on_submit=False):
                c1, c2 = st.columns(2)
                
                with c1:
                    age = st.number_input(t["age"], 0, 20, 5)
                    breed_opts = sorted(list(le_dict.get('Breed species').classes_))
                    breed = st.selectbox(t["breed"], breed_opts)
                    sex_opts = sorted(list(le_dict.get('Sex').classes_))
                    sex = st.selectbox(t["sex"], sex_opts)
                    calvings = st.number_input(t["calvings"], 0, 15, 1)
                    abortion_opts = sorted(list(le_dict.get('Abortion History (Yes No)').classes_))
                    abortion = st.selectbox(t["abortion"], abortion_opts)
                    infertility_opts = sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)').classes_))
                    infertility = st.selectbox(t["infertility"], infertility_opts)
                
                with c2:
                    vaccine_opts = sorted(list(le_dict.get('Brucella vaccination status (Yes No)').classes_))
                    vaccine = st.selectbox(t["vaccination"], vaccine_opts)
                    sample_opts = sorted(list(le_dict.get('Sample Type(Serum Milk)').classes_))
                    sample = st.selectbox(t["sample"], sample_opts)
                    test_opts = sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)').classes_))
                    test = st.selectbox(t["test"], test_opts)
                    retained_opts = sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)').classes_))
                    retained = st.selectbox(t["retained"], retained_opts)
                    disposal_opts = sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)').classes_))
                    disposal = st.selectbox(t["disposal"], disposal_opts)
                
                predict_btn = st.form_submit_button(f"🔬 {t['run_prediction']}", use_container_width=True, type="primary")
            
            # Handle Prediction
            if predict_btn and best_model is not None:
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
                    
                    # Convert all numpy types to native Python types for JSON serialization
                    st.session_state['last_prediction'] = {
                        'result': str(res_label),
                        'confidence': float(probs.max()),
                        'probabilities': [float(p) for p in probs],
                        'classes': [str(c) for c in le_target.classes_],
                        'input_data': {k: (int(v) if isinstance(v, (int, np.integer)) else str(v)) for k, v in input_data.items()}
                    }
                    st.session_state['ai_insight'] = None  # Reset AI insight for new prediction
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        
        # Results Display
        st.markdown("---")
        st.subheader(f"📊 {t['prediction_results']}")
        
        pred = st.session_state.get('last_prediction')
        if pred:
            is_positive = "Positive" in pred['result']
            
            if is_positive:
                st.markdown(f"""
                <div class="prediction-positive">
                    <h3>🔴 POSITIVE</h3>
                    <p>{t['confidence']} {pred['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-negative">
                    <h3>✅ NEGATIVE</h3>
                    <p>{t['confidence']} {pred['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability Chart
            st.subheader(f"📈 {t['probability_dist']}")
            prob_df = pd.DataFrame({
                'Class': pred['classes'],
                'Probability': [p * 100 for p in pred['probabilities']]
            })
            
            fig = go.Figure(data=[
                go.Bar(
                    x=prob_df['Class'],
                    y=prob_df['Probability'],
                    marker_color=['#ef4444' if 'Positive' in c else '#22c55e' for c in prob_df['Class']],
                    text=[f"{v:.1f}%" for v in prob_df['Probability']],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="Probability (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights - Only generate on button click
            st.subheader(f"💡 {t['ai_insights']}")
            
            if st.session_state.get('ai_insight'):
                st.info(st.session_state['ai_insight'])
            else:
                gemini_model = get_gemini_model()
                if gemini_model:
                    if st.button(f"🤖 {t['generate_insights']}", key="gen_insight_btn"):
                        with st.spinner("Generating..."):
                            insight = generate_ai_insight(
                                gemini_model,
                                json.dumps(pred['input_data']),
                                pred['result'],
                                pred['confidence'] * 100,
                                lang
                            )
                            if insight:
                                st.session_state['ai_insight'] = insight
                                st.rerun()
                            else:
                                st.warning("Could not generate insight")
                else:
                    st.caption("AI insights not available (API key not configured)")
        else:
            st.info(t["run_prediction_msg"])
    
    with col_right:
        # AI Chat Assistant
        gemini_model = get_gemini_model()
        if gemini_model:
            st.subheader(f"🤖 {t['vet_assistant']}")
            st.caption(t["vet_subtitle"])
            
            # Chat toggle using checkbox (no rerun issues)
            show_chat = st.checkbox("💬 Open Chat", key="chat_toggle")
            
            if show_chat:
                # Display chat history
                for msg in st.session_state.get('chat_history', []):
                    with st.chat_message(msg['role']):
                        st.write(msg['content'])
                
                # Chat input using form
                with st.form("chat_form", clear_on_submit=True):
                    user_q = st.text_input("Ask a question...", key="chat_q", label_visibility="collapsed")
                    send = st.form_submit_button("Send", use_container_width=True)
                    
                    if send and user_q:
                        st.session_state['chat_history'].append({"role": "user", "content": user_q})
                        response = get_chat_response(gemini_model, user_q, lang)
                        st.session_state['chat_history'].append({"role": "assistant", "content": response})
                        st.rerun()
                
                if st.button("Clear Chat", key="clear_chat"):
                    st.session_state['chat_history'] = []
                    st.rerun()
        
        # Quick Actions
        st.markdown("---")
        st.subheader(f"⚡ {t['quick_actions']}")
        
        with st.expander(f"📄 {t['export_report']}"):
            if pred:
                st.download_button(
                    "Download Report",
                    data=json.dumps(pred, indent=2),
                    file_name="brucellosis_report.json",
                    mime="application/json"
                )
            else:
                st.caption("Run a prediction first")
        
        with st.expander(f"📋 {t['view_guidelines']}"):
            st.markdown("""
            **Brucellosis Prevention:**
            - Vaccinate animals regularly
            - Isolate infected animals
            - Practice proper hygiene
            - Test regularly
            """)
    
    # Footer
    st.markdown("---")
    st.caption("© 2024 BrucellosisAI. Developed for Veterinary Health Solutions.")
