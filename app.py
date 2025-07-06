import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import pickle
import os
import traceback

# Set page config
st.set_page_config(
    page_title="Brucellosis Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# App title
st.markdown('<h1 class="main-header">🩺 Brucellosis Prediction System</h1>', unsafe_allow_html=True)

# Sidebar for model information
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

# Load model and preprocessors from artifacts
@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model and preprocessors from saved files"""
    try:
        # Try multiple possible paths for model artifacts
        possible_paths = [
            "./model_artifacts",
            "./artifacts",
            ".",
            "/app/model_artifacts",
            "/app/artifacts"
        ]
        
        model_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = os.path.join(path, 'best_model.pkl')
                if os.path.exists(model_path):
                    model_dir = path
                    break
        
        if model_dir is None:
            st.error("❌ Model artifacts directory not found!")
            return None, None, None, None, None
        
        # Define file paths
        model_path = os.path.join(model_dir, 'best_model.pkl')
        le_dict_path = os.path.join(model_dir, 'le_dict.pkl')
        le_target_path = os.path.join(model_dir, 'le_target.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')

        # Check if all required files exist
        required_files = [model_path, le_dict_path, le_target_path, feature_names_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ Missing required files: {missing_files}")
            return None, None, None, None, None

        # Load the artifacts
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(le_dict_path, 'rb') as f:
            le_dict = pickle.load(f)
        with open(le_target_path, 'rb') as f:
            le_target = pickle.load(f)
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)

        # Load scaler if it exists (optional)
        scaler = None
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                st.success("✅ Scaler loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Could not load scaler: {e}")

        st.success("✅ Model and preprocessors loaded successfully!")
        return model, le_dict, le_target, scaler, feature_names

    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None, None

# Load model with better error handling
def safe_load_model():
    """Safely load model with fallback options"""
    try:
        return load_model_artifacts()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.info("""
        **Troubleshooting Steps:**
        1. Ensure model artifacts are in the correct directory
        2. Check file permissions
        3. Verify all required files exist:
           - best_model.pkl
           - le_dict.pkl
           - le_target.pkl
           - feature_names.pkl
        """)
        return None, None, None, None, None

# Load model
with st.spinner("Loading model and preprocessors..."):
    model, le_dict, le_target, scaler, feature_names = safe_load_model()

if model is None or le_dict is None or le_target is None or feature_names is None:
    st.error("❌ Failed to load required model components!")
    st.info("Please ensure all model artifacts are properly saved and accessible.")
    st.stop()

# Debug information - show what encoders expect
st.sidebar.markdown("### 🔍 Debug Info")
with st.sidebar.expander("Show Label Encoder Classes"):
    for col, encoder in le_dict.items():
        st.write(f"**{col}:** {list(encoder.classes_)}")

# Main prediction interface
st.markdown('<h2 class="sub-header">🔬 Enter Animal Details</h2>', unsafe_allow_html=True)

# Create input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=20, value=4)
        
        # Breed Species - get actual options from encoder
        if 'Breed species' in le_dict:
            breed_options = list(le_dict['Breed species'].classes_)
            breed = st.selectbox("Breed Species", breed_options, index=0)
        else:
            breed = st.text_input("Breed Species", value="Holstein")
            st.warning("Using default breed options")

        # Sex - use M/F format
        if 'Sex' in le_dict:
            sex_options = list(le_dict['Sex'].classes_)
            sex = st.selectbox("Sex", sex_options, index=0)
        elif ' Sex ' in le_dict:
            sex_options = list(le_dict[' Sex '].classes_)
            sex = st.selectbox("Sex", sex_options, index=0)
        else:
            sex = st.selectbox("Sex", ["F", "M"])

        calvings = st.number_input("Number of Calvings", min_value=0, max_value=10, value=2)

    with col2:
        # Abortion History
        abortion_col = None
        for col in le_dict.keys():
            if 'Abortion' in col:
                abortion_col = col
                break
        
        if abortion_col:
            abortion_options = list(le_dict[abortion_col].classes_)
            abortion_history = st.selectbox("Abortion History", abortion_options, index=0)
        else:
            abortion_history = st.selectbox("Abortion History", ["No", "Yes"])

        # Infertility
        infertility_col = None
        for col in le_dict.keys():
            if 'Infertility' in col:
                infertility_col = col
                break
        
        if infertility_col:
            infertility_options = list(le_dict[infertility_col].classes_)
            infertility = st.selectbox("Infertility/Repeat Breeder", infertility_options, index=0)
        else:
            infertility = st.selectbox("Infertility/Repeat Breeder", ["No", "Yes"])

        # Vaccination Status
        vaccination_col = None
        for col in le_dict.keys():
            if 'vaccination' in col.lower():
                vaccination_col = col
                break
        
        if vaccination_col:
            vaccination_options = list(le_dict[vaccination_col].classes_)
            vaccination = st.selectbox("Brucella Vaccination Status", vaccination_options, index=0)
        else:
            vaccination = st.selectbox("Brucella Vaccination Status", ["No", "Yes"])

        # Sample Type
        sample_col = None
        for col in le_dict.keys():
            if 'Sample' in col:
                sample_col = col
                break
        
        if sample_col:
            sample_options = list(le_dict[sample_col].classes_)
            sample_type = st.selectbox("Sample Type", sample_options, index=0)
        else:
            sample_type = st.selectbox("Sample Type", ["serum", "milk"])

    with col3:
        # Test Type
        test_col = None
        for col in le_dict.keys():
            if 'Test Type' in col:
                test_col = col
                break
        
        if test_col:
            test_options = list(le_dict[test_col].classes_)
            test_type = st.selectbox("Test Type", test_options, index=0)
        else:
            test_type = st.selectbox("Test Type", ["RBPT", "ELISA", "MRT"])

        # Retained Placenta
        retained_col = None
        for col in le_dict.keys():
            if 'Retained' in col or 'Placenta' in col:
                retained_col = col
                break
        
        if retained_col:
            retained_options = list(le_dict[retained_col].classes_)
            retained_placenta = st.selectbox("Retained Placenta/Stillbirth", retained_options, index=0)
        else:
            retained_placenta = st.selectbox("Retained Placenta/Stillbirth", ["No", "Yes", "No Data"])

        # Proper Disposal
        disposal_col = None
        for col in le_dict.keys():
            if 'Disposal' in col:
                disposal_col = col
                break
        
        if disposal_col:
            disposal_options = list(le_dict[disposal_col].classes_)
            disposal = st.selectbox("Proper Disposal of Aborted Fetuses", disposal_options, index=0)
        else:
            disposal = st.selectbox("Proper Disposal of Aborted Fetuses", ["No", "Yes"])

    # Submit button
    submitted = st.form_submit_button("🔍 Predict Brucellosis Status", use_container_width=True)

def safe_encode_value(value, encoder, column_name):
    """Safely encode a value with proper error handling"""
    try:
        if isinstance(value, str):
            value = value.strip()
        encoded = encoder.transform([value])[0]
        return encoded
    except ValueError as e:
        st.warning(f"Unknown value '{value}' for {column_name}. Available options: {list(encoder.classes_)}")
        # Return the most common class (first class as fallback)
        return 0
    except Exception as e:
        st.error(f"Error encoding {column_name}: {str(e)}")
        return 0

def create_input_dataframe(age, breed, sex, calvings, abortion_history, infertility, vaccination, sample_type, test_type, retained_placenta, disposal):
    """Create input dataframe with exact column names from training"""
    
    # Create mapping for user inputs to exact column names
    input_mapping = {}
    
    # Find exact column names from feature_names
    for feature in feature_names:
        if 'Age' in feature:
            input_mapping[feature] = age
        elif 'Breed' in feature:
            input_mapping[feature] = breed
        elif 'Sex' in feature:
            input_mapping[feature] = sex
        elif 'Calvings' in feature:
            input_mapping[feature] = calvings
        elif 'Abortion' in feature:
            input_mapping[feature] = abortion_history
        elif 'Infertility' in feature:
            input_mapping[feature] = infertility
        elif 'vaccination' in feature.lower():
            input_mapping[feature] = vaccination
        elif 'Sample' in feature:
            input_mapping[feature] = sample_type
        elif 'Test Type' in feature:
            input_mapping[feature] = test_type
        elif 'Retained' in feature or 'Placenta' in feature:
            input_mapping[feature] = retained_placenta
        elif 'Disposal' in feature:
            input_mapping[feature] = disposal
    
    # Create DataFrame with exact feature names
    input_df = pd.DataFrame([input_mapping])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    return input_df[feature_names]

if submitted:
    try:
        # Create input DataFrame with exact column names
        input_df = create_input_dataframe(
            age, breed, sex, calvings, abortion_history, 
            infertility, vaccination, sample_type, test_type, 
            retained_placenta, disposal
        )
        
        st.write("**Debug: Input DataFrame columns:**")
        st.write(list(input_df.columns))
        st.write("**Debug: Expected feature names:**")
        st.write(feature_names)
        
        # Encode categorical features safely
        for col in input_df.columns:
            if col in le_dict and input_df[col].dtype == 'object':
                input_df[col] = safe_encode_value(input_df[col].iloc[0], le_dict[col], col)
        
        # Convert all columns to numeric
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                try:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                except:
                    input_df[col] = 0
        
        # Fill any NaN values
        input_df = input_df.fillna(0)
        
        # Scale numerical features if scaler is available
        if scaler is not None:
            # Find numerical columns that need scaling
            numerical_cols = []
            for col in input_df.columns:
                if 'Age' in col or 'Calvings' in col:
                    numerical_cols.append(col)
            
            if numerical_cols:
                # Create a copy for scaling
                input_df_scaled = input_df.copy()
                input_df_scaled[numerical_cols] = scaler.transform(input_df_scaled[numerical_cols])
                prediction_input = input_df_scaled
            else:
                prediction_input = input_df
        else:
            prediction_input = input_df

        # Make prediction
        prediction = model.predict(prediction_input)[0]
        probabilities = model.predict_proba(prediction_input)[0]

        # Convert back to original labels
        predicted_result = le_target.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]

        # Display results
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)

        # Main result box
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

        # Detailed probabilities
        col1, col2, col3 = st.columns(3)
        classes = le_target.classes_
        
        for i, cls in enumerate(classes):
            prob = probabilities[i]
            with [col1, col2, col3][i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{cls}</h3>
                    <h2>{prob:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)

        # Risk assessment
        st.markdown("### 🚨 Risk Assessment")
        
        if predicted_result == "Positive":
            st.error("""
            **HIGH RISK** - Immediate action required:
            - Isolate the animal immediately
            - Consult with veterinarian
            - Follow biosecurity protocols
            - Test other animals in the herd
            """)
        elif predicted_result == "Negative":
            st.success("""
            **LOW RISK** - Continue monitoring:
            - Maintain regular health checks
            - Follow vaccination schedule
            - Monitor for symptoms
            """)
        else:
            st.warning("""
            **MODERATE RISK** - Further investigation needed:
            - Repeat testing recommended
            - Monitor animal closely
            - Consider additional diagnostic tests
            """)

        # Feature importance
        st.markdown("### 📈 Key Risk Factors")
        try:
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.write("Top 10 Most Important Features:")
                st.dataframe(importance_df)
        except:
            st.info("""
            **Key Risk Factors (General):**
            - **Abortion History** - Strong indicator
            - **Age** - Older animals at higher risk
            - **Vaccination Status** - Unvaccinated animals at higher risk
            - **Test Type** - Different sensitivity levels
            - **Sample Type** - Affects accuracy
            """)

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error(traceback.format_exc())
        st.info("Please check your input values and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>🔬 About This System</h4>
    <p>This Brucellosis prediction system uses machine learning to analyze animal health data and predict infection risk.
    The model is trained on veterinary data and uses Extra Trees algorithm for classification.</p>
    <p><strong>Disclaimer:</strong> This tool is for screening purposes only. Always consult with a qualified veterinarian for final diagnosis and treatment decisions.</p>
</div>
""", unsafe_allow_html=True)

# Batch prediction feature
with st.expander("📁 Batch Prediction (Upload CSV)"):
    uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(df_batch.head())
            
            if st.button("Run Batch Predictions"):
                with st.spinner("Processing batch predictions..."):
                    predictions = []
                    
                    for index, row in df_batch.iterrows():
                        try:
                            # Create input DataFrame with exact column names
                            input_df = pd.DataFrame([row.to_dict()])
                            
                            # Ensure all features are present and in correct order
                            for feature in feature_names:
                                if feature not in input_df.columns:
                                    input_df[feature] = 0
                            
                            input_df = input_df[feature_names]
                            
                            # Encode categorical features
                            for col in input_df.columns:
                                if col in le_dict and input_df[col].dtype == 'object':
                                    input_df[col] = safe_encode_value(input_df[col].iloc[0], le_dict[col], col)
                            
                            # Convert to numeric
                            for col in input_df.columns:
                                if input_df[col].dtype == 'object':
                                    try:
                                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                                    except:
                                        input_df[col] = 0
                            
                            input_df = input_df.fillna(0)
                            
                            # Scale if needed
                            if scaler is not None:
                                numerical_cols = []
                                for col in input_df.columns:
                                    if 'Age' in col or 'Calvings' in col:
                                        numerical_cols.append(col)
                                
                                if numerical_cols:
                                    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
                            
                            # Predict
                            pred = model.predict(input_df)[0]
                            prob = model.predict_proba(input_df)[0].max()
                            
                            predictions.append({
                                'Prediction': le_target.inverse_transform([pred])[0],
                                'Confidence': prob
                            })
                            
                        except Exception as e:
                            st.warning(f"Error processing row {index}: {str(e)}")
                            predictions.append({
                                'Prediction': 'Error',
                                'Confidence': 0.0
                            })
                    
                    # Display results
                    results_df = pd.concat([df_batch.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
                    st.write("Batch prediction results:")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="brucellosis_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
