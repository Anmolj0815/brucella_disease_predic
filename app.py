import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import traceback

# ==============================================================================
# Page Configuration and Styling
# ==============================================================================
st.set_page_config(
    page_title="Brucellosis Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .positive-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    }
    .negative-result {
        background: linear-gradient(135deg, #4ecdc4 0%, #2dd4bf 100%);
    }
    .suspect-result {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Sidebar Content
# ==============================================================================
st.markdown('<h1 class="main-header">🩺 Brucellosis Prediction System</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info("""
    **Model:** Extra Trees Classifier
    **Features:** 11 input parameters
    **Classes:** Positive, Negative, Suspect
    **Status:** Ready for predictions
    """)

    st.markdown("### 🔬 About Brucellosis")
    st.write("""
    Brucellosis is a bacterial infection that affects cattle and can be transmitted to humans.
    Early detection is crucial for livestock health management.
    """)

# ==============================================================================
# Model and Artifact Loading
# ==============================================================================
@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model and preprocessors from saved files."""
    try:
        # Define file paths relative to the script location
        model_dir = "." 
        model_path = os.path.join(model_dir, 'best_model.pkl')
        le_dict_path = os.path.join(model_dir, 'le_dict.pkl')
        le_target_path = os.path.join(model_dir, 'le_target.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')

        # Check if all required files exist
        required_files = [model_path, le_dict_path, le_target_path, feature_names_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ Missing required artifact files: {', '.join(missing_files)}")
            st.info("Please ensure 'best_model.pkl', 'le_dict.pkl', 'le_target.pkl', and 'feature_names.pkl' are in the same directory as the script.")
            return None, None, None, None, None

        # Load the artifacts
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(le_dict_path, 'rb') as f: le_dict = pickle.load(f)
        with open(le_target_path, 'rb') as f: le_target = pickle.load(f)
        with open(feature_names_path, 'rb') as f: feature_names = pickle.load(f)

        # Load scaler if it exists (it's optional)
        scaler = None
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                st.success("✅ Scaler loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Could not load optional scaler.pkl file: {e}")

        st.success("✅ Model and preprocessors loaded successfully!")
        return model, le_dict, le_target, scaler, feature_names

    except Exception as e:
        st.error(f"❌ An error occurred while loading model artifacts: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None, None

# Load model and artifacts
with st.spinner("Loading model and preprocessors..."):
    model, le_dict, le_target, scaler, feature_names = load_model_artifacts()

# Stop the app if loading failed
if not all([model, le_dict, le_target, feature_names]):
    st.error("❌ Failed to load required model components. The application cannot continue.")
    st.stop()

# Display debug info in the sidebar
with st.sidebar.expander("🔍 Show Label Encoder Classes"):
    for col, encoder in le_dict.items():
        st.write(f"**{col.strip()}:** {list(encoder.classes_)}")

# ==============================================================================
# User Input Form
# ==============================================================================
st.markdown('<h2 class="sub-header">🔬 Enter Animal Details</h2>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=30, value=5, step=1)
        breed = st.text_input("Breed Species", value="Holstein", help="e.g., Holstein, Jersey, etc.")
        sex = st.text_input("Sex", value="F", help="Enter 'F' for Female or 'M' for Male.")
        calvings = st.number_input("Number of Calvings", min_value=0, max_value=15, value=2, step=1)

    with col2:
        abortion_history = st.text_input("Abortion History", value="No", help="Enter 'Yes' or 'No'.")
        infertility = st.text_input("Infertility/Repeat Breeder", value="No", help="Enter 'Yes' or 'No'.")
        vaccination = st.text_input("Brucella Vaccination Status", value="No", help="Enter 'Yes' or 'No'.")
        sample_type = st.text_input("Sample Type", value="serum", help="e.g., serum, milk.")

    with col3:
        test_type = st.text_input("Test Type", value="RBPT", help="e.g., RBPT, ELISA, MRT.")
        retained_placenta = st.text_input("Retained Placenta/Stillbirth", value="No", help="Enter 'Yes' or 'No'.")
        disposal = st.text_input("Proper Disposal of Aborted Fetuses", value="No", help="Enter 'Yes' or 'No'.")

    submitted = st.form_submit_button("🔍 Predict Brucellosis Status", use_container_width=True)

# ==============================================================================
# Prediction Logic
# ==============================================================================
def safe_encode_value(value, encoder, column_name):
    """Safely encode a value, trying different capitalizations before falling back."""
    # Standardize the input by stripping whitespace
    value_str = str(value).strip()
    
    # Try common capitalizations
    possible_values = [value_str, value_str.lower(), value_str.upper(), value_str.capitalize()]
    
    for v in possible_values:
        try:
            return encoder.transform([v])[0]
        except ValueError:
            continue # Try the next format
            
    st.warning(f"Unknown value '{value}' for '{column_name}'. Available options: {list(encoder.classes_)}. Using a default value.")
    return 0 # Fallback to the most common class (encoded as 0)

if submitted:
    try:
        # 1. Map UI inputs to clean feature names.
        user_input_map = {
            'Age': age,
            'Breed species': breed,
            'Sex': sex,
            'Number of Calvings': calvings,
            'Abortion History': abortion_history,
            'Infertility/Repeat Breeder': infertility,
            'Brucella Vaccination Status': vaccination,
            'Sample Type': sample_type,
            'Test Type': test_type,
            'Retained Placenta/Stillbirth': retained_placenta,
            'Proper Disposal of Aborted Fetuses': disposal
        }
        
        # 2. Build the input dictionary using the exact (potentially messy) feature names from training.
        input_data = {messy_name: user_input_map.get(messy_name.strip(), 0) for messy_name in feature_names}

        # 3. Create the DataFrame with the exact column names the model expects.
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # 4. Encode categorical features.
        for col in input_df.columns:
            if col in le_dict and input_df[col].dtype == 'object':
                original_value = input_df[col].iloc[0]
                encoded_value = safe_encode_value(original_value, le_dict[col], col)
                input_df[col] = encoded_value
        
        input_df = input_df.fillna(0).astype(float)

        # 5. Apply scaling if a scaler was loaded.
        if scaler:
            scaler_features = getattr(scaler, 'feature_names_in_', None)
            if scaler_features is not None:
                if set(scaler_features).issubset(input_df.columns):
                    input_df[list(scaler_features)] = scaler.transform(input_df[list(scaler_features)])
                else:
                    st.warning("Scaler features don't match input. Skipping scaling.")
            else:
                 st.warning("Scaler has no saved feature names. Skipping scaling.")

        # 6. Ensure final column order is correct before prediction.
        input_df = input_df[feature_names]

        # 7. Make the prediction.
        prediction_encoded = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # 8. Decode the prediction back to a human-readable label.
        predicted_result = le_target.inverse_transform([prediction_encoded])[0]
        confidence = probabilities[prediction_encoded]

        # ==============================================================================
        # Display Results
        # ==============================================================================
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)

        result_class = {
            "Positive": "positive-result",
            "Negative": "negative-result",
            "Suspect": "suspect-result"
        }.get(predicted_result, "prediction-box")

        st.markdown(f"""
        <div class="prediction-box {result_class}">
            <h2>🎯 Predicted Result: {predicted_result}</h2>
            <h3>Confidence: {confidence:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Display detailed probabilities for each class
        prob_cols = st.columns(len(le_target.classes_))
        for i, cls_encoded in enumerate(le_target.classes_):
            cls_label = le_target.inverse_transform([cls_encoded])[0]
            prob = probabilities[i]
            with prob_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{cls_label}</h3>
                    <h2>{prob:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Display risk assessment based on the result
        st.markdown("### 🚨 Risk Assessment")
        if predicted_result == "Positive":
            st.error("**HIGH RISK:** Immediate action required. Isolate the animal, consult a veterinarian, and test the herd.")
        elif predicted_result == "Negative":
            st.success("**LOW RISK:** Continue routine monitoring and maintain biosecurity protocols.")
        else: # Suspect
            st.warning("**MODERATE RISK:** Further investigation is needed. Re-testing is recommended.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error(traceback.format_exc())
        st.info("Please check your input values and ensure they match the expected formats.")

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>🔬 About This System</h4>
    <p>This Brucellosis prediction system uses a machine learning model (Extra Trees Classifier) to analyze animal health data and predict infection risk. It is intended as a screening tool to aid in herd management.</p>
    <p><strong>Disclaimer:</strong> This tool is for informational purposes only. Always consult with a qualified veterinarian for a definitive diagnosis and treatment decisions.</p>
</div>
""", unsafe_allow_html=True)
