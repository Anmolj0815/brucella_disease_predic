import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import json
from passlib.hash import bcrypt
import base64

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
USERS_FILE = MODEL_ARTIFACTS_DIR + 'users.json'
VETERINARY_IMAGE_PATH = MODEL_ARTIFACTS_DIR + 'veterinary.jpg'

# --- CSS STYLING ---
def inject_custom_css():
    """Inject custom CSS for beautiful background and styling"""
    st.markdown("""
    <style>
    /* Main background with animated dots */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        position: relative;
        overflow: hidden;
    }
    
    /* Animated background dots */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 20%, rgba(255,255,255,0.1) 2px, transparent 2px),
            radial-gradient(circle at 80% 80%, rgba(255,255,255,0.1) 2px, transparent 2px),
            radial-gradient(circle at 40% 40%, rgba(255,255,255,0.05) 1px, transparent 1px),
            radial-gradient(circle at 60% 60%, rgba(255,255,255,0.05) 1px, transparent 1px);
        background-size: 50px 50px, 75px 75px, 25px 25px, 35px 35px;
        animation: float 20s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    /* Login page styling */
    .login-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 2rem auto;
        max-width: 800px;
    }
    
    /* Main app content styling */
    .main-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
    }
    
    /* Title styling */
    .main-title {
        color: #4a5568;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Welcome text styling */
    .welcome-text {
        color: #2d3748;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Image container */
    .image-container {
        text-align: center;
        margin: 2rem 0;
    }
    
    .image-container img {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        max-width: 100%;
        height: auto;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Prediction result styling */
    .prediction-success {
        background: linear-gradient(45deg, #48bb78, #38a169);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .prediction-info {
        background: linear-gradient(45deg, #4299e1, #3182ce);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD USER CREDENTIALS ---
users = {}
try:
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    st.sidebar.success("🔒 User credentials loaded successfully!")
except FileNotFoundError:
    st.sidebar.error(f"❌ User credentials file not found at '{USERS_FILE}'. Please create it.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Error loading user credentials: {e}")
    st.stop()

# --- INITIALIZE SESSION STATE FOR LOGIN ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- LOAD MODEL ARTIFACTS ---
try:
    with open(MODEL_ARTIFACTS_DIR + 'best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'le_dict.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'le_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    st.sidebar.success("✅ All model components loaded successfully!")

except FileNotFoundError as e:
    st.sidebar.error(f"❌ Required model file not found: {e}. Please ensure all .pkl files are in the '{MODEL_ARTIFACTS_DIR}' directory (or update the path).")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Error loading model components: {e}")
    st.stop()

# --- FUNCTION TO LOAD AND ENCODE IMAGE ---
def load_image_as_base64(image_path):
    """Load image and convert to base64 for embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"⚠️ Image not found at {image_path}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading image: {e}")
        return None

# --- AUTHENTICATION FUNCTION ---
def login_page():
    # Load and display veterinary image
    img_base64 = load_image_as_base64(VETERINARY_IMAGE_PATH)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-title">🐂 Brucellosis Prediction System</h1>', unsafe_allow_html=True)
    
    # Display image if available
    if img_base64:
        st.markdown(f'''
        <div class="image-container">
            <img src="data:image/jpeg;base64,{img_base64}" alt="Veterinary Image" style="max-width: 600px;">
        </div>
        ''', unsafe_allow_html=True)
    
    # Welcome text
    st.markdown('''
    <div class="welcome-text">
        <p>Welcome to the Advanced Brucellosis Prediction System</p>
        <p>A comprehensive tool for veterinary professionals to predict and analyze Brucellosis in livestock</p>
        <p>Please login to access the prediction system</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Login form in sidebar
    st.sidebar.subheader("🔐 Login")
    email = st.sidebar.text_input("📧 Email")
    password = st.sidebar.text_input("🔒 Password", type="password")

    if st.sidebar.button("🚀 Login"):
        if email in users:
            if bcrypt.verify(password, users[email]):
                st.session_state['logged_in'] = True
                st.session_state['username'] = email
                st.sidebar.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid email or password.")
                st.session_state['logged_in'] = False
        else:
            st.sidebar.error("❌ Invalid email or password.")
            st.session_state['logged_in'] = False

    st.sidebar.markdown("---")
    st.sidebar.info("Please enter your registered email and password to access the app.")

# --- MAIN APP LOGIC ---
st.set_page_config(page_title="Brucellosis Prediction App", layout="wide")

# Inject custom CSS
inject_custom_css()

if not st.session_state['logged_in']:
    login_page()
else:
    # --- MAIN APP CONTENT ---
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">🐂 Brucellosis Prediction Model</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="welcome-text">Welcome, <strong>{st.session_state["username"]}</strong>! Enter the animal\'s details to predict its Brucellosis status.</div>', unsafe_allow_html=True)

    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()
    
    st.sidebar.markdown("---")

    # Get unique categories for dropdowns
    unique_breeds = sorted(list(le_dict.get('Breed species', LabelEncoder()).classes_))
    unique_sex = sorted(list(le_dict.get('Sex', LabelEncoder()).classes_))
    unique_abortion_history = sorted(list(le_dict.get('Abortion History (Yes No)', LabelEncoder()).classes_))
    unique_infertility = sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)', LabelEncoder()).classes_))
    unique_vaccination_status = sorted(list(le_dict.get('Brucella vaccination status (Yes No)', LabelEncoder()).classes_))
    unique_sample_type = sorted(list(le_dict.get('Sample Type(Serum Milk)', LabelEncoder()).classes_))
    unique_test_type = sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)', LabelEncoder()).classes_))
    unique_retained_placenta = sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)', LabelEncoder()).classes_))
    unique_disposal = sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)', LabelEncoder()).classes_))

    def predict_single_case(input_dict, model, le_dict, le_target, scaler, feature_names):
        """Predict a single case with robust error handling for encoding."""

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Strip spaces from input_df columns
        input_df.columns = input_df.columns.str.strip()

        # Pre-process 'Breed species' content
        if 'Breed species' in input_df.columns:
            input_df['Breed species'] = input_df['Breed species'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

        # Encode categorical features
        for col in input_df.columns:
            if col in le_dict and input_df.dtypes.get(col) == 'object':
                try:
                    input_df.loc[:, col] = le_dict.get(col).transform(input_df.loc[:, col])
                except ValueError as e:
                    st.error(f"❌ Error encoding column '{col}': The input value '{input_dict.get(col)}' is not a known category. Known categories: {list(le_dict.get(col).classes_)}")
                    return None

        # Ensure all expected features are present, fill with 0 if not
        for col in feature_names:
            if col not in input_df.columns:
                input_df.loc[:, col] = 0

        # Reorder columns to match training data's feature order
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Decide if scaling is needed based on the type of the 'best_model'
        model_requires_scaling = isinstance(model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))

        if model_requires_scaling:
            input_df_scaled = scaler.transform(input_df)
        else:
            input_df_scaled = input_df.values

        # Predict
        pred_class = model.predict(input_df_scaled)[0]
        pred_prob = model.predict_proba(input_df_scaled)[0]

        # Convert back to original labels
        predicted_result = le_target.inverse_transform([pred_class])[0]
        confidence = pred_prob.max()

        return {
            'predicted_class': predicted_result,
            'confidence': confidence,
            'probabilities': dict(zip(le_target.classes_, pred_prob))
        }

    st.sidebar.header("🔍 Input Features")

    # Collect user input using columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Age (Years)", 0, 20, 5)
        breed_species = st.selectbox("🐄 Breed/Species", options=unique_breeds)
        sex = st.selectbox("⚥ Sex", options=unique_sex)
        calvings = st.slider("🐮 Calvings", 0, 15, 1)
        abortion_history = st.selectbox("📋 Abortion History", options=unique_abortion_history)

    with col2:
        infertility_rb = st.selectbox("🔄 Infertility/Repeat Breeder", options=unique_infertility)
        vaccination_status = st.selectbox("💉 Brucella Vaccination Status", options=unique_vaccination_status)
        sample_type = st.selectbox("🧪 Sample Type", options=unique_sample_type)
        test_type = st.selectbox("🔬 Test Type", options=unique_test_type)
        retained_placenta = st.selectbox("⚠️ Retained Placenta/Stillbirth", options=unique_retained_placenta)
        proper_disposal = st.selectbox("🗑️ Proper Disposal of Aborted Fetuses", options=unique_disposal)

    input_data = {
        'Age': age,
        'Breed species': breed_species,
        'Sex': sex,
        'Calvings': calvings,
        'Abortion History (Yes No)': abortion_history,
        'Infertility Repeat breeder(Yes No)': infertility_rb,
        'Brucella vaccination status (Yes No)': vaccination_status,
        'Sample Type(Serum Milk)': sample_type,
        'Test Type (RBPT ELISA MRT)': test_type,
        'Retained Placenta Stillbirth(Yes No No Data)': retained_placenta,
        'Proper Disposal of Aborted Fetuses (Yes No)': proper_disposal
    }

    st.subheader("📊 Provided Input:")
    st.json(input_data)

    if st.button("🎯 Predict Brucellosis Status"):
        st.subheader("📈 Prediction Results:")
        with st.spinner('🔄 Making prediction...'):
            prediction_output = predict_single_case(
                input_data, best_model, le_dict, le_target, scaler, feature_names
            )

            if prediction_output:
                st.markdown(f'<div class="prediction-success">🎯 Predicted Result: {prediction_output["predicted_class"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-info">📊 Confidence: {prediction_output["confidence"]:.2%}</div>', unsafe_allow_html=True)

                st.write("---")
                st.subheader("📋 Class-wise Probabilities:")
                prob_df = pd.DataFrame.from_dict(prediction_output['probabilities'], orient='index', columns=['Probability'])
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df.style.format("{:.2%}"))

                # Visualizing probabilities
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=prob_df.index, y=prob_df['Probability'], palette='viridis', ax=ax)
                ax.set_title("Predicted Class Probabilities", fontsize=16, fontweight='bold')
                ax.set_ylabel("Probability", fontsize=12)
                ax.set_xlabel("Brucellosis Status", fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("❌ Failed to make a prediction. Please check the input values and error messages above.")

    st.markdown("---")
    st.markdown('<div class="welcome-text">🩺 Developed with ❤️ for Veterinary Health</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
