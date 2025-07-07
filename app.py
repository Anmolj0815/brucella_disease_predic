import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import traceback
# Removed confusion_matrix, matplotlib.pyplot, seaborn as they are not used with best_model.pkl directly
# If you want to add them back, you'll need to save/load y_test as a separate artifact.

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

# Define the path to your artifacts directory
ARTIFACTS_DIR = 'model_artifacts'

# Function to load model and preprocessors from artifacts
@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model and preprocessors from saved files"""
    try:
        model_dir = None
        # Prioritize local relative path, then common deployment paths
        possible_paths = [
            "./", # Current directory
            os.path.join(".", ARTIFACTS_DIR), # Expected for local setup
            os.path.join("/app", ARTIFACTS_DIR), # Common for Streamlit Cloud
            os.path.join("/mount/src/your_repo_name", ARTIFACTS_DIR) # Another common Streamlit Cloud path
        ]

        for path in possible_paths:
            # Look for best_model.pkl directly within these paths
            full_model_path = os.path.join(path, 'best_model.pkl')
            # st.write(f"DEBUG: Checking path: {full_model_path}") # Debugging
            if os.path.exists(full_model_path):
                model_dir = path
                # st.write(f"DEBUG: Found model directory: {model_dir}") # Debugging
                break

        if model_dir is None:
            st.error(f"❌ `best_model.pkl` not found! Please ensure it's in the '{ARTIFACTS_DIR}' directory.")
            return None, None, None, None, None, None

        # Define full file paths
        best_model_path = os.path.join(model_dir, 'best_model.pkl')
        le_dict_path = os.path.join(model_dir, 'le_dict.pkl')
        le_target_path = os.path.join(model_dir, 'le_target.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
        df_clean_path = os.path.join(model_dir, 'df_clean.csv')

        # Check if all required files exist
        required_files = [best_model_path, le_dict_path, le_target_path, feature_names_path, df_clean_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            st.error(f"❌ Missing required files in '{model_dir}' directory: {missing_files}")
            return None, None, None, None, None, None

        # Load the artifacts
        with open(best_model_path, 'rb') as f:
            model = pickle.load(f)

        with open(le_dict_path, 'rb') as f:
            le_dict = pickle.load(f)
        with open(le_target_path, 'rb') as f:
            le_target = pickle.load(f)
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
            # CRITICAL: Ensure feature names are stripped after loading
            feature_names = [f.strip() for f in feature_names]

        # Load df_clean
        df_clean = pd.read_csv(df_clean_path)
        df_clean.columns = df_clean.columns.str.strip() # Ensure df_clean columns are stripped

        # Load scaler if it exists
        scaler = None
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                st.success("✅ Scaler loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Could not load scaler: {e}")

        st.success("✅ Model and preprocessors loaded successfully!")
        return model, le_dict, le_target, scaler, feature_names, df_clean

    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None, None, None

# Load model and preprocessors
with st.spinner("Loading model and preprocessors..."):
    model, le_dict, le_target, scaler, feature_names, df_clean = load_model_artifacts()

if model is None or le_dict is None or le_target is None or feature_names is None or df_clean is None:
    st.error("❌ Failed to load required model components or data! Please check the error messages above.")
    st.info("Ensure the `model_artifacts` directory contains `best_model.pkl`, `le_dict.pkl`, `le_target.pkl`, `feature_names.pkl`, `df_clean.csv`, and optionally `scaler.pkl`.")
    st.stop() # Stop the app if essential components are not loaded

# Sidebar for model information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Model:** {model.__class__.__name__} (Loaded from best_model.pkl)
    **Features:** {len(feature_names)} input parameters
    **Classes:** {', '.join(le_target.classes_)}
    **Status:** Ready for predictions
    """)

    st.markdown("### 🔬 About Brucellosis")
    st.write("""
    Brucellosis is a bacterial infection that affects cattle and can be transmitted to humans.
    Early detection is crucial for livestock health management.
    """)

# Helper function for safe encoding
def safe_encode_value(value, encoder, column_name):
    """Safely encode a value with proper error handling and fallback."""
    try:
        # CRITICAL: Apply strip().title() to the input value for consistency
        if isinstance(value, str):
            cleaned_value = value.strip().title()
        else:
            cleaned_value = value # For numerical values, no cleaning needed

        # Check if the cleaned value is in the encoder's known classes
        if cleaned_value not in encoder.classes_:
            st.warning(f"Unknown value '{value}' for {column_name}. Using fallback (first class: '{encoder.classes_[0]}').")
            # Fallback: return the encoding of the first class, or a common/default class
            return encoder.transform([encoder.classes_[0]])[0]

        encoded = encoder.transform([cleaned_value])[0]
        return encoded
    except Exception as e:
        st.error(f"Error encoding {column_name} with value '{value}': {str(e)}. Using fallback (first class).")
        # Fallback in case of any other encoding error
        return encoder.transform([encoder.classes_[0]])[0]

# Main prediction interface
st.markdown('<h2 class="sub-header">🔬 Enter Animal Details</h2>', unsafe_allow_html=True)

# Create input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=20, value=4)

        # CRITICAL: Populate selectbox options by cleaning df_clean values
        breed_options = sorted([str(x).strip().title() for x in df_clean['Breed species'].unique().tolist() if pd.notna(x)])
        breed = st.selectbox("Breed Species", breed_options)

        sex_options = sorted([str(x).strip().title() for x in df_clean['Sex'].unique().tolist() if pd.notna(x)])
        sex = st.selectbox("Sex", sex_options)

        calvings = st.number_input("Number of Calvings", min_value=0, max_value=10, value=2)

    with col2:
        abortion_history_options = sorted([str(x).strip().title() for x in df_clean['Abortion History (Yes No)'].unique().tolist() if pd.notna(x)])
        abortion_history = st.selectbox("Abortion History", abortion_history_options)

        infertility_options = sorted([str(x).strip().title() for x in df_clean['Infertility Repeat breeder(Yes No)'].unique().tolist() if pd.notna(x)])
        infertility = st.selectbox("Infertility/Repeat Breeder", infertility_options)

        vaccination_options = sorted([str(x).strip().title() for x in df_clean['Brucella vaccination status (Yes No)'].unique().tolist() if pd.notna(x)])
        vaccination = st.selectbox("Brucella Vaccination Status", vaccination_options)

        sample_options = sorted([str(x).strip().title() for x in df_clean['Sample Type(Serum Milk)'].unique().tolist() if pd.notna(x)])
        sample_type = st.selectbox("Sample Type", sample_options)

    with col3:
        test_options = sorted([str(x).strip().title() for x in df_clean['Test Type (RBPT ELISA MRT)'].unique().tolist() if pd.notna(x)])
        test_type = st.selectbox("Test Type", test_options)

        retained_options = sorted([str(x).strip().title() for x in df_clean['Retained Placenta Stillbirth(Yes No No Data)'].unique().tolist() if pd.notna(x)])
        retained_placenta = st.selectbox("Retained Placenta/Stillbirth", retained_options)

        disposal_options = sorted([str(x).strip().title() for x in df_clean['Proper Disposal of Aborted Fetuses (Yes No)'].unique().tolist() if pd.notna(x)])
        disposal = st.selectbox("Proper Disposal of Aborted Fetuses", disposal_options)

    # Submit button
    submitted = st.form_submit_button("🔍 Predict Brucellosis Status", use_container_width=True)

if submitted:
    try:
        # Prepare input data: Keys here MUST match the *stripped* feature names
        input_data = {
            'Age': age,
            'Breed species': breed,
            'Sex': sex,
            'Calvings': calvings,
            'Abortion History (Yes No)': abortion_history,
            'Infertility Repeat breeder(Yes No)': infertility,
            'Brucella vaccination status (Yes No)': vaccination,
            'Sample Type(Serum Milk)': sample_type,
            'Test Type (RBPT ELISA MRT)': test_type,
            'Retained Placenta Stillbirth(Yes No No Data)': retained_placenta,
            'Proper Disposal of Aborted Fetuses (Yes No)': disposal
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        # No need to strip columns here, as keys in input_data are already stripped to match feature_names

        # Encode categorical features safely
        for col_name_stripped in input_df.columns:
            if col_name_stripped in le_dict and input_df[col_name_stripped].dtype == 'object':
                input_df[col_name_stripped] = safe_encode_value(input_df[col_name_stripped].iloc[0], le_dict[col_name_stripped], col_name_stripped)

        # Ensure all features are present and in the correct order as per 'feature_names'
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0 # Default for new columns

        # Reorder columns to match training data
        input_df = input_df[feature_names]

        # Scale numerical features if scaler is available
        if scaler is not None:
            numerical_cols_stripped = ['Age', 'Calvings']
            existing_numerical_cols = [col for col in numerical_cols_stripped if col in input_df.columns]
            if existing_numerical_cols:
                input_df[existing_numerical_cols] = scaler.transform(input_df[existing_numerical_cols])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Convert back to original labels
        predicted_result = le_target.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] # Confidence for the predicted class

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
            prob_index = np.where(le_target.classes_ == cls)[0]
            if prob_index.size > 0: # Check if class exists in probabilities
                prob = probabilities[prob_index[0]]
            else:
                prob = 0.0 # Default if class not found for some reason

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
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)

                st.write("Top 10 Most Important Features:")
                st.dataframe(importance_df, hide_index=True)
            else:
                st.info("""
                **Key Risk Factors (General - based on common veterinary knowledge):**
                - **Abortion History** - Strong indicator
                - **Age** - Older animals might be at higher risk
                - **Vaccination Status** - Unvaccinated animals at higher risk
                - **Test Type** - Different sensitivity levels for various tests
                - **Sample Type** - Can affect test accuracy
                """)
        except Exception as e:
            st.error(f"Error displaying feature importance: {str(e)}")


    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error(traceback.format_exc())
        st.info("Please check your input values and ensure all required model artifacts are loaded correctly.")

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>🔬 About This System</h4>
    <p>This Brucellosis prediction system uses machine learning to analyze animal health data and predict infection risk.
    The model is trained on veterinary data and uses an ensemble algorithm for classification.</p>
    <p><strong>Disclaimer:</strong> This tool is for screening purposes only. Always consult with a qualified veterinarian for final diagnosis and treatment decisions.</p>
</div>
""", unsafe_allow_html=True)

# Batch prediction feature
with st.expander("📁 Batch Prediction (Upload CSV)"):
    uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type="csv")

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Data preview of uploaded file:")
            st.dataframe(df_batch.head())

            if st.button("Run Batch Predictions"):
                with st.spinner("Processing batch predictions..."):
                    predictions = []

                    # CRITICAL: Ensure batch DataFrame columns are stripped to match expected features
                    df_batch.columns = df_batch.columns.str.strip()

                    for index, row in df_batch.iterrows():
                        try:
                            # Create a single-row DataFrame from the current row
                            input_dict_batch = row.to_dict()
                            # CRITICAL: Strip keys in the dictionary before creating DataFrame
                            input_dict_batch_cleaned = {k.strip(): v for k, v in input_dict_batch.items()}
                            input_df_batch = pd.DataFrame([input_dict_batch_cleaned])
                            # Columns are already stripped from input_df_batch due to cleaned dict keys

                            # Encode categorical features
                            for col_name_stripped in input_df_batch.columns:
                                if col_name_stripped in le_dict and input_df_batch[col_name_stripped].dtype == 'object':
                                    input_df_batch[col_name_stripped] = safe_encode_value(input_df_batch[col_name_stripped].iloc[0], le_dict[col_name_stripped], col_name_stripped)

                            # Ensure all features are present
                            for col in feature_names:
                                if col not in input_df_batch.columns:
                                    input_df_batch[col] = 0 # Default for missing columns

                            # Reorder columns to match training data
                            input_df_batch = input_df_batch[feature_names]

                            # Scale if needed
                            if scaler is not None:
                                numerical_cols_stripped = ['Age', 'Calvings']
                                existing_numerical_cols = [col for col in numerical_cols_stripped if col in input_df_batch.columns]
                                if existing_numerical_cols:
                                    input_df_batch[existing_numerical_cols] = scaler.transform(input_df_batch[existing_numerical_cols])

                            # Predict
                            pred = model.predict(input_df_batch)[0]
                            prob = model.predict_proba(input_df_batch)[0]

                            predictions.append({
                                'Prediction': le_target.inverse_transform([pred])[0],
                                'Confidence': prob[pred] # Confidence of the predicted class
                            })

                        except Exception as e:
                            st.warning(f"Error processing row {index + 1}: {str(e)}. This row will be marked as 'Error'.")
                            predictions.append({
                                'Prediction': 'Error',
                                'Confidence': 0.0
                            })

                    # Display results
                    results_df = pd.concat([df_batch.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
                    st.write("Batch prediction results:")
                    st.dataframe(results_df, use_container_width=True)

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
