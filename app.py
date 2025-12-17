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
        
        # List all available models and use the first one
        try:
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            if available_models:
                # Use the first available model
                model_to_use = available_models[0]
                gemini_model = genai.GenerativeModel(model_name=model_to_use)
                ai_enabled = True
            else:
                st.sidebar.warning("⚠️ No models found with generateContent support.")
                
        except Exception as list_error:
            # Fallback: try common model names without testing
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            for model_name in model_names:
                try:
                    gemini_model = genai.GenerativeModel(model_name=model_name)
                    ai_enabled = True
                    break
                except:
                    continue
            
            if not ai_enabled:
                st.sidebar.warning("⚠️ Could not initialize Gemini. AI advice disabled.")
            
    except Exception as e:
        st.sidebar.error(f"AI Setup Error: {e}")
else:
    st.sidebar.warning("⚠️ Gemini API Key not found in Streamlit Secrets. AI advice disabled.")

# --- TRANSLATIONS & AUTOMATED PROMPT TEMPLATES ---
translations = {
    "English": {
        "welcome": "Welcome to Brucellosis Prediction App",
        "title": "🐂 Brucellosis Prediction Model",
        "user_greet": "Welcome, **{}**!",
        "input_header": "Input Features",
        "age": "Age (Years)", "breed": "Breed/Species", "sex": "Sex",
        "calvings": "Calvings", "abortion": "Abortion History (Yes/No)",
        "infertility": "Infertility/Repeat Breeder (Yes/No)",
        "vaccination": "Brucella Vaccination Status (Yes/No)",
        "sample": "Sample Type (Serum/Milk)", "test": "Test Type (RBPT/ELISA/MRT)",
        "retained": "Retained Placenta/Stillbirth", "disposal": "Proper Disposal of Aborted Fetuses (Yes No)",
        "predict_btn": "Predict Brucellosis Status",
        "results_header": "Prediction Results:", "pred_res": "**Predicted Status:**",
        "conf": "**Confidence Score:**", "prob_header": "Probability Analysis:",
        "chart_title": "Class Distribution", "logout": "Logout", "login_sub": "Login",
        "ai_advice_header": "🤖 AI Veterinary Consultation",
        "ai_loading": "Analyzing data and generating suggestions...",
        "system_prompt": "You are a senior veterinary expert. Analyzing animal data: {}. Prediction Result: {}. Confidence: {}%. If result is Positive, strongly advise immediate isolation and confirmatory lab testing (RBPT/ELISA). Provide 3-4 clear, actionable steps for the farmer in English.",
        "chatbot_button": "💬 Ask AI Chikitsak",
        "chatbot_title": "🩺 AI Chikitsak - Your Veterinary Assistant",
        "chatbot_subtitle": "Ask me anything about Brucellosis, milk safety, and animal health",
        "chat_placeholder": "Ask your question about Brucellosis, symptoms, prevention, milk safety...",
        "chat_system": "You are 'AI Chikitsak' (AI Doctor), an expert veterinary consultant specializing in Brucellosis and dairy animal health. Answer questions about: Brucellosis disease, symptoms in animals, transmission, prevention, vaccination, milk safety, treatment, diagnosis tests (RBPT/ELISA/MRT), farm biosecurity, and general cattle/buffalo health. Provide clear, practical advice in English. Keep answers concise (3-5 sentences) unless detailed explanation is requested."
    },
    "Hindi": {
        "welcome": "ब्रुसेलोसिस भविष्यवाणी ऐप में आपका स्वागत है",
        "title": "🐂 ब्रुसेलोसिस भविष्यवाणी मॉडल",
        "user_greet": "आपका स्वागत है, **{}**!",
        "input_header": "इनपुट विशेषताएं",
        "age": "आयु (वर्ष)", "breed": "नस्ल/प्रजाति", "sex": "लिंग",
        "calvings": "बछड़े की संख्या", "abortion": "गर्भपात का इतिहास",
        "infertility": "बांझपन", "vaccination": "टीकाकरण की स्थिति",
        "sample": "नमूना प्रकार", "test": "परीक्षण प्रकार",
        "retained": "जेर रुकना/मृत प्रसव", "disposal": "भ्रूण का निपटान",
        "predict_btn": "भविष्यवाणी करें",
        "results_header": "परिणाम:", "pred_res": "**अनुमानित स्थिति:**",
        "conf": "**भरोसा (Confidence):**", "prob_header": "संभावना विश्लेषण:",
        "chart_title": "संभावना चार्ट", "logout": "लॉगआउट", "login_sub": "लॉगिन",
        "ai_advice_header": "🤖 AI पशु चिकित्सक सलाह",
        "ai_loading": "डेटा का विश्लेषण और सुझाव तैयार किए जा रहे हैं...",
        "system_prompt": "आप एक वरिष्ठ पशु चिकित्सा विशेषज्ञ हैं। पशु डेटा: {}. भविष्यवाणी परिणाम: {}. भरोसा: {}%. यदि परिणाम पॉजिटिव है, तो तुरंत पशु को अलग करने (Isolation) और लैब टेस्टिंग (RBPT/ELISA) की सलाह दें। किसान के लिए हिंदी में 3-4 स्पष्ट और व्यावहारिक सुझाव दें।",
        "chatbot_button": "💬 AI चिकित्सक से पूछें",
        "chatbot_title": "🩺 AI चिकित्सक - आपका पशु चिकित्सा सहायक",
        "chatbot_subtitle": "ब्रुसेलोसिस, दूध की सुरक्षा और पशु स्वास्थ्य के बारे में कुछ भी पूछें",
        "chat_placeholder": "ब्रुसेलोसिस, लक्षण, रोकथाम, दूध की सुरक्षा के बारे में अपना सवाल पूछें...",
        "chat_system": "आप 'AI चिकित्सक' हैं, एक विशेषज्ञ पशु चिकित्सा सलाहकार जो ब्रुसेलोसिस और डेयरी पशु स्वास्थ्य में विशेषज्ञता रखते हैं। इन विषयों पर सवालों के जवाब दें: ब्रुसेलोसिस रोग, पशुओं में लक्षण, संचरण, रोकथाम, टीकाकरण, दूध की सुरक्षा, उपचार, निदान परीक्षण (RBPT/ELISA/MRT), फार्म बायोसिक्योरिटी, और सामान्य गाय/भैंस स्वास्थ्य। हिंदी में स्पष्ट, व्यावहारिक सलाह दें। जवाब संक्षिप्त (3-5 वाक्य) रखें जब तक विस्तृत स्पष्टीकरण न मांगा जाए।"
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
        
        # Create email
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
        
        # Send email via Gmail SMTP
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
    
    # Check expiry (10 minutes)
    if time.time() - st.session_state['otp_timestamp'] > 600:
        return False, "OTP expired. Please request a new one."
    
    if entered_otp == st.session_state['otp_code']:
        return True, "OTP verified successfully!"
    else:
        return False, "Invalid OTP. Please try again."

def connect_to_google_sheet():
    """Connect to Google Sheets"""
    try:
        # Define the scope
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        
        # Get credentials from Streamlit secrets
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        
        # Open the Google Sheet
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
        
        # Check if headers exist, if not add them
        try:
            headers = sheet.row_values(1)
            if not headers:
                sheet.append_row(['Email', 'Name', 'Phone', 'Location', 'Registration Date'])
        except:
            sheet.append_row(['Email', 'Name', 'Phone', 'Location', 'Registration Date'])
        
        # Add new user data
        registration_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        sheet.append_row([email, name, phone, location, registration_date])
        
        st.success(f"✅ User data saved to Google Sheet successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error saving to Google Sheet: {e}")
        return False

def register_user(email, password, name, phone, location):
    """Register new user in JSON and Google Sheet"""
    try:
        # Load existing users from JSON
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
        else:
            users = {}
        
        # Check if user already exists
        if email in users:
            return False, "User already exists!"
        
        # Hash password and save to JSON
        users[email] = pbkdf2_sha256.hash(password)
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
        
        # Save to Google Sheet
        if save_user_to_google_sheet(email, name, phone, location):
            return True, "Registration successful!"
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

# --- UI LOGIC ---
st.set_page_config(page_title="Brucella AI Predictor", layout="wide")
selected_lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
t = translations[selected_lang]

if not st.session_state['logged_in']:
    st.title(t["welcome"])
    
    # Tabs for Login and Register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader(t["login_sub"])
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            try:
                with open(USERS_FILE, 'r') as f: users = json.load(f)
                if email in users and pbkdf2_sha256.verify(password, users[email]):
                    st.session_state.update(logged_in=True, username=email)
                    st.rerun()
                else: st.error("Invalid credentials")
            except: st.error("User database not found.")
    
    with tab2:
        st.subheader("Register New Account")
        
        if not st.session_state['otp_sent']:
            # Step 1: Collect user details
            reg_name = st.text_input("Full Name", key="reg_name")
            reg_email = st.text_input("Email", key="reg_email")
            reg_phone = st.text_input("Phone Number", key="reg_phone")
            reg_location = st.text_input("Location (City/Village)", key="reg_location")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            if st.button("Send OTP"):
                if not all([reg_name, reg_email, reg_phone, reg_location, reg_password]):
                    st.error("Please fill all fields")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    # Check if user already exists
                    try:
                        if os.path.exists(USERS_FILE):
                            with open(USERS_FILE, 'r') as f:
                                users = json.load(f)
                            if reg_email in users:
                                st.error("User already exists!")
                            else:
                                # Generate and send OTP
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
                                    st.success(f"✅ OTP sent to {reg_email}. Please check your email.")
                                    st.rerun()
                        else:
                            # First user - send OTP
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
                                st.success(f"✅ OTP sent to {reg_email}. Please check your email.")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        else:
            # Step 2: Verify OTP
            st.info(f"📧 OTP sent to {st.session_state['pending_user_data']['email']}")
            st.caption("Please enter the 6-digit OTP sent to your email (valid for 10 minutes)")
            
            entered_otp = st.text_input("Enter OTP", max_chars=6, key="otp_input")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Verify OTP"):
                    if entered_otp:
                        is_valid, message = verify_otp(entered_otp)
                        if is_valid:
                            # Register user
                            user_data = st.session_state['pending_user_data']
                            success, reg_message = register_user(
                                user_data['email'],
                                user_data['password'],
                                user_data['name'],
                                user_data['phone'],
                                user_data['location']
                            )
                            if success:
                                st.success("🎉 " + reg_message + " Please login now.")
                                # Reset OTP session
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
                        st.error("Please enter OTP")
            
            with col2:
                if st.button("Resend OTP"):
                    otp = generate_otp()
                    if send_otp_email(st.session_state['pending_user_data']['email'], otp):
                        st.session_state['otp_code'] = otp
                        st.session_state['otp_timestamp'] = time.time()
                        st.success("✅ New OTP sent!")
                        st.rerun()
            
            if st.button("← Back to Registration"):
                st.session_state['otp_sent'] = False
                st.session_state['otp_code'] = None
                st.session_state['otp_timestamp'] = None
                st.session_state['pending_user_data'] = None
                st.rerun()
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
        # 1. PRE-PROCESSING & ENCODING
        input_df = pd.DataFrame([input_data])
        
        # Ensure categorical encoding using exactly the same logic as training
        for col in input_df.columns:
            if col in le_dict and input_df[col].dtype == 'object':
                input_df[col] = le_dict[col].transform(input_df[col])

        # Align with training data order
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # 2. SCALING & PREDICTION
        is_linear = isinstance(best_model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))
        processed = scaler.transform(input_df) if is_linear else input_df.values
        
        try:
            pred_idx = best_model.predict(processed)[0]
            probs = best_model.predict_proba(processed)[0]
            res_label = le_target.inverse_transform([pred_idx])[0]
            conf_score = probs.max()

            # 3. DISPLAY RESULTS
            st.markdown("---")
            st.subheader(t["results_header"])
            ui_res = "पॉजिटिव (Positive)" if (selected_lang == "Hindi" and "Positive" in res_label) else \
                     "नेगेटिव (Negative)" if (selected_lang == "Hindi" and "Negative" in res_label) else res_label
            
            st.success(f"{t['pred_res']} {ui_res}")
            st.info(f"{t['conf']} {conf_score:.2%}")

            # 4. AI CONSULTATION (FIXED CALL)
            if ai_enabled:
                st.subheader(t["ai_advice_header"])
                with st.spinner(t["ai_loading"]):
                    try:
                        auto_prompt = t["system_prompt"].format(json.dumps(input_data), res_label, round(conf_score*100, 2))
                        # FIX: Using the generated object to call content generation
                        response = gemini_model.generate_content(auto_prompt)
                        st.markdown(f"> {response.text}")
                    except Exception as ai_e:
                        st.error(f"AI Generation Error: {ai_e}")

            # 5. VISUALIZATION
            st.write("---")
            st.subheader(t["prob_header"])
            prob_df = pd.DataFrame({'Probability': probs}, index=le_target.classes_)
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(x=prob_df.index, y=prob_df['Probability'], palette='viridis', ax=ax)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    st.markdown("---")
    st.markdown("Developed with ❤️ for Veterinary Health")
    
    # --- FLOATING CHATBOT BUTTON ---
    if ai_enabled:
        st.markdown("""
        <style>
        .chatbot-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 999;
            background-color: #4CAF50;
            color: white;
            padding: 15px 25px;
            border-radius: 50px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            cursor: pointer;
            border: none;
        }
        .chatbot-button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button(t["chatbot_button"], key="chatbot_toggle"):
            st.session_state['show_chatbot'] = not st.session_state['show_chatbot']
        
        # --- CHATBOT MODAL ---
        if st.session_state['show_chatbot']:
            st.markdown("---")
            st.subheader(t["chatbot_title"])
            st.caption(t["chatbot_subtitle"])
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for i, msg in enumerate(st.session_state['chat_history']):
                    if msg['role'] == 'user':
                        st.markdown(f"**🧑 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🩺 AI Chikitsak:** {msg['content']}")
            
            # Chat input
            user_question = st.text_input(t["chat_placeholder"], key="chat_input")
            
            col_send, col_clear = st.columns([4, 1])
            with col_send:
                if st.button("Send", key="send_btn") and user_question:
                    # Add user message
                    st.session_state['chat_history'].append({"role": "user", "content": user_question})
                    
                    # Get AI response
                    with st.spinner("AI Chikitsak is thinking..."):
                        try:
                            full_prompt = f"{t['chat_system']}\n\nUser Question: {user_question}"
                            response = gemini_model.generate_content(full_prompt)
                            ai_response = response.text
                            st.session_state['chat_history'].append({"role": "assistant", "content": ai_response})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Chat Error: {e}")
            
            with col_clear:
                if st.button("Clear Chat", key="clear_btn"):
                    st.session_state['chat_history'] = []
                    st.rerun()
