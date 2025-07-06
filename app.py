import streamlit as st
import pandas as pd
import numpy as np
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
    **Model:** Extra Trees Classifier (or best performing model from training)
    **Features:** 11 input parameters
    **Classes:** Positive, Negative, Suspect
    **Status:** Ready for predictions
    """)

    st.markdown("### 🔬 About Brucellosis")
    st.write("""
    Brucellosis is a bacterial infection that affects cattle and can be transmitted to humans.
    Early detection is crucial for livestock health management.
    """)

# --- Model Loading ---
@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model and preprocessors from saved files."""
    try:
        # Define possible paths for model artifacts
        possible_paths = [
            "./model_artifacts", # Common for local development
            "./artifacts",       # Another common path
            ".",                 # Current directory
            "/app/model_artifacts", # For deployment environments
            "/app/artifacts"
        ]
        
        model_dir = None
        for path in possible_paths:
            # Check if 'best_model.pkl' exists in any of the possible paths
            model_path_candidate = os.path.join(path, 'best_model.pkl')
            if os.path.exists(model_path_candidate):
                model_dir = path
                break
        
        if model_dir is None:
            st.error("❌ Model artifacts directory not found! Please ensure 'best_model.pkl' and other files are in a recognized directory.")
            return None, None, None, None, None
        
        # Construct full file paths once model_dir is found
        model_path = os.path.join(model_dir, 'best_model.pkl')
        le_dict_path = os.path.join(model_dir, 'le_dict.pkl')
        le_target_path = os.path.join(model_dir, 'le_target.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')

        # Check if all required files exist
        required_files = [model_path, le_dict_path, le_target_path, feature_names_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"❌ Missing required model artifact files: {', '.join(missing_files)}")
            return None, None, None, None, None

        # Load the artifacts using pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(le_dict_path, 'rb') as f:
            le_dict = pickle.load(f)
        with open(le_target_path, 'rb') as f:
            le_target = pickle.load(f)
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)

        # Load scaler if it exists (it's optional based on model type)
        scaler = None
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                st.success("✅ Scaler loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Could not load scaler from '{scaler_path}': {e}. Predictions might be inaccurate if scaling is required.")

        st.success("✅ Model and preprocessors loaded successfully!")
        return model, le_dict, le_target, scaler, feature_names

    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading model artifacts: {str(e)}")
        st.error(traceback.format_exc()) # Display full traceback for debugging
        return None, None, None, None, None

# Safely load model and preprocessors
with st.spinner("Loading model and preprocessors..."):
    model, le_dict, le_target, scaler, feature_names = load_model_artifacts()

# Stop the app if essential components failed to load
if model is None or le_dict is None or le_target is None or feature_names is None:
    st.error("❌ Application cannot proceed due to missing model components. Please check the artifact files.")
    st.stop()

# Debug information for label encoder classes
st.sidebar.markdown("### 🔍 Debug Info")
with st.sidebar.expander("Show Label Encoder Classes"):
    for col, encoder in le_dict.items():
        st.write(f"**{col.strip()}:** {list(encoder.classes_)}")

# --- Helper Functions for Prediction ---

def safe_encode_value(value, encoder, column_name):
    """
    Safely encodes a categorical value using a LabelEncoder.
    Handles values not seen during training by defaulting to 0 and issuing a warning.
    """
    if not isinstance(value, str):
        value = str(value) # Ensure value is a string for stripping

    value = value.strip() # Clean input value
    
    # Check if the value is in the encoder's known classes
    if value not in encoder.classes_:
        st.warning(f"Input value '{value}' for '{column_name}' not seen during training. "
                   f"Available options: {list(encoder.classes_)}. Defaulting to 0.")
        return 0 # Default to the first encoded class (usually 0) or a specific fallback
    try:
        encoded = encoder.transform([value])[0]
        return encoded
    except Exception as e:
        st.error(f"Error encoding '{column_name}' with value '{value}': {str(e)}. Defaulting to 0.")
        return 0

def create_input_dataframe(age, breed, sex, calvings, abortion_history, infertility, vaccination, sample_type, test_type, retained_placenta, disposal, feature_names):
    """
    Creates a Pandas DataFrame for prediction, ensuring column names and order
    exactly match the features the model was trained on.
    """
    # Initialize a dictionary with all expected feature names from 'feature_names'
    # This ensures all features are present and provides a placeholder for each.
    input_data_dict = {feature: None for feature in feature_names} 

    # Map user inputs to the exact feature names as found in feature_names.
    # It's crucial that these keys match the strings in feature_names precisely,
    # including any leading/trailing spaces if they exist in the original feature_names.
    # We will iterate through feature_names to find the best match for user inputs.
    
    # Direct mappings based on common sense and typical feature names
    # The actual feature names are loaded from feature_names.pkl
    # We will try to match based on substrings to be more robust,
    # but exact matches are preferred.

    # Age and Calvings are numerical, directly assign
    for f_name in feature_names:
        if f_name.strip().lower() == 'age':
            input_data_dict[f_name] = age
        elif f_name.strip().lower() == 'calvings' or 'number of calvings' in f_name.strip().lower():
            input_data_dict[f_name] = calvings
        
    # Categorical features - find the exact key from feature_names
    # This loop ensures we use the exact key (e.g., ' Sex ') if that's what's in feature_names
    feature_mapping = {
        'breed species': breed,
        'sex': sex,
        'abortion history': abortion_history,
        'infertility repeat breeder': infertility,
        'brucella vaccination status': vaccination,
        'sample type': sample_type,
        'test type': test_type,
        'retained placenta stillbirth': retained_placenta,
        'proper disposal of aborted fetuses': disposal
    }

    for f_name in feature_names:
        stripped_f_name = f_name.strip().lower()
        for map_key, user_value in feature_mapping.items():
            if map_key in stripped_f_name: # Use 'in' for partial matching, better than strict equality
                input_data_dict[f_name] = user_value
                break # Found a match, move to next feature_name

    # Create DataFrame from the dictionary.
    input_df = pd.DataFrame([input_data_dict])
    
    # Crucially, reindex the DataFrame to ensure the columns are in the exact order
    # as expected by the model during fit. This step is vital for preventing the
    # "feature names should match" error.
    input_df = input_df[feature_names]
    
    return input_df

# --- Main Prediction Interface ---
st.markdown('<h2 class="sub-header">🔬 Enter Animal Details</h2>', unsafe_allow_html=True)

# Create input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=20, value=4)
        breed = st.text_input("Breed Species (e.g., Holstein, Jersey)", value="Holstein")
        sex = st.text_input("Sex (e.g., F, M)", value="F")
        calvings = st.number_input("Number of Calvings", min_value=0, max_value=10, value=2)

    with col2:
        abortion_history = st.text_input("Abortion History (e.g., Yes, No)", value="No")
        infertility = st.text_input("Infertility/Repeat Breeder (e.g., Yes, No)", value="No")
        vaccination = st.text_input("Brucella Vaccination Status (e.g., Yes, No)", value="No")
        sample_type = st.text_input("Sample Type (e.g., serum, milk)", value="serum")

    with col3:
        test_type = st.text_input("Test Type (e.g., RBPT, ELISA, MRT)", value="ELISA")
        retained_placenta = st.text_input("Retained Placenta/Stillbirth (e.g., Yes, No, No Data)", value="No Data")
        disposal = st.text_input("Proper Disposal of Aborted Fetuses (e.g., Yes, No)", value="Yes")

    # Submit button
    submitted = st.form_submit_button("🔍 Predict Brucellosis Status", use_container_width=True)

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
            # Find the correct key in le_dict based on the feature name
            le_key_found = None
            for key in le_dict.keys():
                if key.strip().lower() == col.strip().lower(): # Case-insensitive and space-agnostic match
                    le_key_found = key
                    break
            
            if le_key_found and input_df[col].dtype == 'object':
                original_value = input_df[col].iloc[0]
                encoded_value = safe_encode_value(original_value, le_dict[le_key_found], col)
                input_df[col] = encoded_value
                st.write(f"Encoded {col}: '{original_value}' -> {encoded_value}")
        
        # Convert any remaining object columns to numeric (fallback for unexpected types)
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                try:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Could not convert column '{col}' to numeric: {e}. Setting to 0.")
                    input_df[col] = 0
        
        # Fill any NaN values that might have resulted from 'coerce' during numeric conversion
        input_df = input_df.fillna(0)
        
        st.write("**Debug: Final input data before scaling (if applicable):**")
        st.write(input_df)
        st.write("**Debug: Final input data dtypes:**")
        st.write(input_df.dtypes)

        # Apply scaling if a scaler is present and the model benefits from it
        # The original script applies scaling conditionally based on model type
        # Assuming 'model' is the best_model loaded, we need to check its type
        model_type_name = str(type(model))
        needs_scaling = any(m_type in model_type_name for m_type in ["MLPClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier"])

        if scaler is not None and needs_scaling:
            if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                scaler_features = list(scaler.feature_names_in_)
                # Ensure all features expected by the scaler are present in input_df
                if set(scaler_features).issubset(set(input_df.columns)):
                    # Create a temporary DataFrame with only scaler-expected features in correct order
                    scaler_input_df = input_df[scaler_features]
                    scaled_values = scaler.transform(scaler_input_df)
                    
                    # Update the original input_df with scaled values
                    input_df[scaler_features] = scaled_values
                    st.write("**Debug: Applied scaling successfully using scaler's feature names.**")
                else:
                    st.warning("Scaler features do not fully match input features. Skipping scaling for potentially missing columns.")
            else:
                st.warning("Scaler does not have 'feature_names_in_'. Attempting to scale all numerical columns.")
                numerical_cols = input_df.select_dtypes(include=np.number).columns.tolist()
                if numerical_cols:
                    try:
                        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
                        st.write(f"**Debug: Scaled numerical columns:** {numerical_cols}")
                    except Exception as e:
                        st.warning(f"Could not apply scaling to numerical columns: {str(e)}")
        elif scaler is None and needs_scaling:
            st.warning("Model requires scaling but scaler was not loaded. Prediction accuracy may be affected.")


        # Final check to ensure DataFrame has exactly the right columns in the right order for prediction
        # This is a critical step to prevent the ValueError
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

        # --- Display Results ---
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)

        # Main result box with dynamic styling
        result_class = {
            "Positive": "positive-result",
            "Negative": "negative-result",
            "Suspect": "suspect-result"
        }.get(predicted_result, "prediction-box") # Default if result is unexpected

        st.markdown(f"""
        <div class="prediction-box {result_class}">
            <h2>🎯 Predicted Result: {predicted_result}</h2>
            <h3>Confidence: {confidence:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Detailed probabilities
        st.markdown("### Class-wise Probabilities:")
        cols_prob = st.columns(len(le_target.classes_))
        classes = le_target.classes_
        
        for i, cls in enumerate(classes):
            prob = probabilities[i]
            with cols_prob[i]:
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
        else: # Suspect
            st.warning("""
            **MODERATE RISK** - Further investigation needed:
            - Repeat testing recommended
            - Monitor animal closely
            - Consider additional diagnostic tests
            """)

        # Feature importance (if available in the model)
        st.markdown("### 📈 Key Risk Factors")
        try:
            if hasattr(model, 'feature_importances_') and feature_names:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.write("Top 10 Most Important Features:")
                st.dataframe(importance_df)
            else:
                st.info("Feature importance not available for this model type.")
        except Exception as e:
            st.info(f"Could not display feature importance: {e}")
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
        st.info("Please check your input values and try again. Ensure categorical inputs match expected values from training data (e.g., 'Yes'/'No', 'serum'/'milk').")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>🔬 About This System</h4>
    <p>This Brucellosis prediction system uses machine learning to analyze animal health data and predict infection risk.
    The model is trained on veterinary data and uses Extra Trees algorithm for classification.</p>
    <p><strong>Disclaimer:</strong> This tool is for screening purposes only. Always consult with a qualified veterinarian for final diagnosis and treatment decisions.</p>
</div>
""", unsafe_allow_html=True)

# --- Batch Prediction Feature ---
with st.expander("📁 Batch Prediction (Upload CSV)"):
    uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(df_batch.head())
            
            if st.button("Run Batch Predictions", key="batch_predict_button"):
                with st.spinner("Processing batch predictions..."):
                    predictions_list = []
                    
                    # Ensure batch DataFrame columns are aligned with expected feature names
                    # Add missing columns with default values (e.g., 0)
                    for feature in feature_names:
                        if feature not in df_batch.columns:
                            df_batch[feature] = 0 # Default value for missing features in batch file
                    
                    # Ensure columns are in the correct order for the model
                    df_batch_processed = df_batch[feature_names].copy()

                    for index, row in df_batch_processed.iterrows():
                        try:
                            # Convert row to dictionary to pass to create_input_dataframe logic
                            # Note: For batch, the 'create_input_dataframe' logic is slightly different
                            # as we already have a DataFrame row. We need to apply encoding and scaling.
                            
                            # Create a single-row DataFrame for the current input
                            input_df_row = pd.DataFrame([row.to_dict()])
                            input_df_row = input_df_row[feature_names] # Ensure order

                            # Encode categorical features for the current row
                            for col in input_df_row.columns:
                                le_key_found = None
                                for key in le_dict.keys():
                                    if key.strip().lower() == col.strip().lower():
                                        le_key_found = key
                                        break

                                if le_key_found and input_df_row[col].dtype == 'object':
                                    input_df_row[col] = safe_encode_value(input_df_row[col].iloc[0], le_dict[le_key_found], col)
                            
                            # Convert to numeric
                            for col in input_df_row.columns:
                                if input_df_row[col].dtype == 'object':
                                    try:
                                        input_df_row[col] = pd.to_numeric(input_df_row[col], errors='coerce')
                                    except:
                                        input_df_row[col] = 0
                            
                            input_df_row = input_df_row.fillna(0)
                            
                            # Scale if needed for the current row
                            if scaler is not None and needs_scaling:
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
                            prob = model.predict_proba(input_df_row)[0].max() # Get max probability for the predicted class
                            
                            predictions_list.append({
                                'Prediction': le_target.inverse_transform([pred])[0],
                                'Confidence': prob
                            })
                            
                        except Exception as e:
                            st.warning(f"Error processing row {index} in batch: {str(e)}")
                            predictions_list.append({
                                'Prediction': 'Error',
                                'Confidence': 0.0
                            })
                    
                    # Display results
                    results_df = pd.concat([df_batch.reset_index(drop=True), pd.DataFrame(predictions_list)], axis=1)
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
            st.error(f"Error processing uploaded file: {str(e)}")
            st.error(traceback.format_exc())
