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
        st.write(f"**{col.strip()}:** {list(encoder.classes_)}") # Strip spaces for display

# Main prediction interface
st.markdown('<h2 class="sub-header">🔬 Enter Animal Details</h2>', unsafe_allow_html=True)

# Create input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=20, value=4)
        
        # Breed Species - provide text input
        breed = st.text_input("Breed Species (e.g., Holstein, Jersey, etc.)", value="Holstein")
        
        # Sex - provide text input
        sex = st.text_input("Sex (F for Female, M for Male)", value="F")

        calvings = st.number_input("Number of Calvings", min_value=0, max_value=10, value=2)

    with col2:
        # Abortion History - provide text input
        abortion_history = st.text_input("Abortion History (e.g., Yes, No)", value="No")

        # Infertility - provide text input
        infertility = st.text_input("Infertility/Repeat Breeder (e.g., Yes, No)", value="No")

        # Vaccination Status - provide text input
        vaccination = st.text_input("Brucella Vaccination Status (e.g., Yes, No)", value="No")

        # Sample Type - provide text input
        sample_type = st.text_input("Sample Type (e.g., serum, milk)", value="serum")

    with col3:
        # Test Type - provide text input
        test_type = st.text_input("Test Type (e.g., RBPT, ELISA, MRT)", value="RBPT")

        # Retained Placenta - provide text input
        retained_placenta = st.text_input("Retained Placenta/Stillbirth (e.g., Yes, No, No Data)", value="No")

        # Proper Disposal - provide text input
        disposal = st.text_input("Proper Disposal of Aborted Fetuses (e.g., Yes, No)", value="Yes")

    # Submit button
    submitted = st.form_submit_button("🔍 Predict Brucellosis Status", use_container_width=True)

def safe_encode_value(value, encoder, column_name):
    """Safely encode a value with proper error handling"""
    if not isinstance(value, str):
        value = str(value) # Ensure value is a string for stripping

    value = value.strip()
    if value not in encoder.classes_:
        st.warning(f"Unknown value '{value}' for {column_name}. Available options: {list(encoder.classes_)}. Defaulting to 0.")
        return 0 # Default to 0 or another sensible fallback
    try:
        encoded = encoder.transform([value])[0]
        return encoded
    except Exception as e:
        st.error(f"Error encoding {column_name} ('{value}'): {str(e)}. Defaulting to 0.")
        return 0 # Fallback in case of unexpected encoding errors

def create_input_dataframe(age, breed, sex, calvings, abortion_history, infertility, vaccination, sample_type, test_type, retained_placenta, disposal, feature_names):
    """Create input dataframe with exact column names and order from training"""
    
    # Create a dictionary to hold input data, ensuring all expected features are present
    input_data_dict = {}
    for feature in feature_names:
        input_data_dict[feature] = None # Initialize with None or a default value

    # Map user inputs to exact column names
    # Use exact feature names as keys for direct assignment
    # This assumes 'feature_names' contains the exact names used during model training
    input_data_dict['Age'] = age
    input_data_dict['Breed species'] = breed
    input_data_dict['Sex'] = sex
    input_data_dict['Number of Calvings'] = calvings
    input_data_dict['Abortion History'] = abortion_history
    input_data_dict['Infertility/Repeat Breeder'] = infertility
    input_data_dict['Brucella Vaccination Status'] = vaccination
    input_data_dict['Sample Type'] = sample_type
    input_data_dict['Test Type'] = test_type
    input_data_dict['Retained Placenta/Stillbirth'] = retained_placenta
    input_data_dict['Proper Disposal of Aborted Fetuses'] = disposal

    # Create DataFrame from the dictionary, ensuring column order matches feature_names
    input_df = pd.DataFrame([input_data_dict])
    
    # Reindex to ensure the exact order of columns as in feature_names
    input_df = input_df[feature_names]
    
    return input_df


if submitted:
    try:
        # Create input DataFrame with exact column names and order
        input_df = create_input_dataframe(
            age, breed, sex, calvings, abortion_history, 
            infertility, vaccination, sample_type, test_type, 
            retained_placenta, disposal, feature_names
        )
        
        st.write("**Debug: Input DataFrame columns after creation:**")
        st.write(list(input_df.columns))
        st.write("**Debug: Expected feature names:**")
        st.write(feature_names)
        st.write("**Debug: Columns match expected order and names (should be True):**")
        st.write(list(input_df.columns) == feature_names)
        
        # Encode categorical features safely
        for col in input_df.columns:
            # Find the correct key in le_dict, handling potential leading/trailing spaces
            le_key = None
            for key in le_dict.keys():
                if key.strip() == col.strip():
                    le_key = key
                    break
            
            if le_key and input_df[col].dtype == 'object':
                original_value = input_df[col].iloc[0]
                encoded_value = safe_encode_value(original_value, le_dict[le_key], col)
                input_df[col] = encoded_value
                st.write(f"Encoded {col}: '{original_value}' -> {encoded_value}")
        
        # Convert any remaining object columns (should mostly be handled by encoding) to numeric if possible
        # This is a fallback for columns that might have been missed or have mixed types
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                try:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Could not convert column '{col}' to numeric: {e}. Setting to 0.")
                    input_df[col] = 0
        
        # Fill any NaN values after potential numeric conversion
        input_df = input_df.fillna(0)
        
        st.write("**Debug: Final input data before scaling (if applicable):**")
        st.write(input_df)
        st.write("**Debug: Final input data dtypes:**")
        st.write(input_df.dtypes)

        # Handle scaling carefully
        if scaler is not None:
            if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                scaler_features = list(scaler.feature_names_in_)
                st.write(f"**Debug: Scaler expects features:** {scaler_features}")
                
                # Check if all scaler features are present in the input DataFrame
                if set(scaler_features).issubset(set(input_df.columns)):
                    # Create a temporary DataFrame with only scaler-expected features in correct order
                    scaler_input_df = input_df[scaler_features]
                    scaled_values = scaler.transform(scaler_input_df)
                    
                    # Update the original input_df with scaled values
                    input_df[scaler_features] = scaled_values
                    
                    st.write("**Debug: Applied scaling successfully.**")
                else:
                    st.warning("Scaler features do not fully match input features. Skipping scaling for potentially missing columns.")
            else:
                st.warning("Scaler does not have 'feature_names_in_'. Attempting to scale all numerical columns.")
                # Fallback: identify numerical columns and scale them
                numerical_cols = input_df.select_dtypes(include=np.number).columns.tolist()
                if numerical_cols:
                    try:
                        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
                        st.write(f"**Debug: Scaled numerical columns:** {numerical_cols}")
                    except Exception as e:
                        st.warning(f"Could not apply scaling to numerical columns: {str(e)}")

        # Ensure DataFrame has exactly the right columns in the right order for prediction
        input_df = input_df[feature_names]
        
        st.write("**Debug: Final DataFrame ready for prediction (after all processing):**")
        st.write(input_df)
        st.write("**Debug: Final DataFrame shape:**")
        st.write(input_df.shape)
        st.write("**Debug: Final DataFrame columns (should match feature_names exactly):**")
        st.write(list(input_df.columns))

        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

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
        st.info("Please check your input values and try again. Ensure categorical inputs match expected values from training data.")

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
                    
                    # Align batch DataFrame columns with expected feature names
                    # Add missing columns with default values (e.g., 0 or mode)
                    for feature in feature_names:
                        if feature not in df_batch.columns:
                            df_batch[feature] = 0 # Or a more appropriate default/imputation
                    
                    # Ensure columns are in the correct order for the model
                    df_batch = df_batch[feature_names]

                    for index, row in df_batch.iterrows():
                        try:
                            # Create input DataFrame for the current row, already aligned
                            input_df_row = pd.DataFrame([row.to_dict()])
                            input_df_row = input_df_row[feature_names] # Re-ensure order

                            # Encode categorical features for the current row
                            for col in input_df_row.columns:
                                le_key = None
                                for key in le_dict.keys():
                                    if key.strip() == col.strip():
                                        le_key = key
                                        break
                                
                                if le_key and input_df_row[col].dtype == 'object':
                                    input_df_row[col] = safe_encode_value(input_df_row[col].iloc[0], le_dict[le_key], col)
                            
                            # Convert to numeric
                            for col in input_df_row.columns:
                                if input_df_row[col].dtype == 'object':
                                    try:
                                        input_df_row[col] = pd.to_numeric(input_df_row[col], errors='coerce')
                                    except:
                                        input_df_row[col] = 0
                            
                            input_df_row = input_df_row.fillna(0)
                            
                            # Scale if needed for the current row
                            if scaler is not None:
                                if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                                    scaler_features = list(scaler.feature_names_in_)
                                    if set(scaler_features).issubset(set(input_df_row.columns)):
                                        input_df_row[scaler_features] = scaler.transform(input_df_row[scaler_features])
                                else:
                                    numerical_cols = input_df_row.select_dtypes(include=np.number).columns.tolist()
                                    if numerical_cols:
                                        input_df_row[numerical_cols] = scaler.transform(input_df_row[numerical_cols])

                            # Predict
                            pred = model.predict(input_df_row)[0]
                            prob = model.predict_proba(input_df_row)[0].max()
                            
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
