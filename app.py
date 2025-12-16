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
        "system_prompt": "You are a senior veterinary expert. Analyzing animal data: {}. Prediction Result: {}. Confidence: {}%. If result is Positive, strongly advise immediate isolation and confirmatory lab testing (RBPT/ELISA). Provide 3-4 clear, actionable steps for the farmer in English."
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
        "system_prompt": "आप एक वरिष्ठ पशु चिकित्सा विशेषज्ञ हैं। पशु डेटा: {}. भविष्यवाणी परिणाम: {}. भरोसा: {}%. यदि परिणाम पॉजिटिव है, तो तुरंत पशु को अलग करने (Isolation) और लैब टेस्टिंग (RBPT/ELISA) की सलाह दें। किसान के लिए हिंदी में 3-4 स्पष्ट और व्यावहारिक सुझाव दें।"
    }
}

# --- MODEL LOADING ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

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
    st.sidebar.subheader(t["login_sub"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        try:
            with open(USERS_FILE, 'r') as f: users = json.load(f)
            if email in users and pbkdf2_sha256.verify(password, users[email]):
                st.session_state.update(logged_in=True, username=email)
                st.rerun()
            else: st.sidebar.error("Invalid credentials")
        except: st.sidebar.error("User database not found.")
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
