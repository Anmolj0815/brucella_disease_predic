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
    **Model:** Extra Trees Classifier (loaded from artifacts)
    **Features:** 11 input parameters
    **Classes:** Positive, Negative, Suspect
    **Accuracy:** (Based on training in notebook)
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

    model_dir = "./model_artifacts"
    model_path = os.path.join(model_dir, 'best_model.pkl')
    le_dict_path = os.path.join(model_dir, 'le_dict.pkl')
    le_target_path = os.path.join(model_dir, 'le_target.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')

    if os.path.exists(model_path) and os.path.exists(le_dict_path) and \
       os.path.exists(le_target_path) and os.path.exists(feature_names_path):
        st.write("Loading pre-trained model and preprocessors...")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(le_dict_path, 'rb') as f:
                le_dict = pickle.load(f)
            with open(le_target_path, 'rb') as f:
                le_target = pickle.load(f)
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)

            # Load scaler if it exists
            scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                st.success("✅ Scaler loaded successfully!")


            st.success("✅ Pre-trained model and preprocessors loaded successfully!")
            return model, le_dict, le_target, scaler, feature_names
        except Exception as e:
            st.error(f"❌ Error loading model artifacts: {e}")
            return None, None, None, None, None
    else:
        st.error("❌ Model artifacts not found! Please ensure 'model_artifacts' directory exists with all required files.")
        st.info("Refer to the notebook steps to save the model artifacts.")
        return None, None, None, None, None


# Load model
with st.spinner("Loading model and preprocessors..."):
    model, le_dict, le_target, scaler, feature_names = load_model_artifacts()

if model is None or le_dict is None or le_target is None or feature_names is None:
    st.stop() # Stop the app if loading failed

# Main prediction interface
st.markdown('<h2 class="sub-header">🔬 Enter Animal Details</h2>', unsafe_allow_html=True)

# Create input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=20, value=4)

        # Use values from the loaded le_dict for dropdowns
        if 'Breed species' in le_dict:
            breed_options = list(le_dict['Breed species'].classes_)
            breed = st.selectbox("Breed Species", breed_options)
        else:
             breed = st.text_input("Breed Species")
             st.warning("Breed species options could not be loaded. Inputting as text.")


        if 'Sex' in le_dict:
             sex_options = list(le_dict['Sex'].classes_)
             sex = st.selectbox("Sex", sex_options)
        else:
             sex = st.text_input("Sex")
             st.warning("Sex options could not be loaded. Inputting as text.")


        calvings = st.number_input("Number of Calvings", min_value=0, max_value=10, value=2)

    with col2:
        if 'Abortion History (Yes No)' in le_dict:
            abortion_options = list(le_dict['Abortion History (Yes No)'].classes_)
            abortion_history = st.selectbox("Abortion History", abortion_options)
        else:
             abortion_history = st.text_input("Abortion History")
             st.warning("Abortion History options could not be loaded. Inputting as text.")


        if 'Infertility Repeat breeder(Yes No)' in le_dict:
            infertility_options = list(le_dict['Infertility Repeat breeder(Yes No)'].classes_)
            infertility = st.selectbox("Infertility/Repeat Breeder", infertility_options)
        else:
            infertility = st.text_input("Infertility/Repeat Breeder")
            st.warning("Infertility/Repeat Breeder options could not be loaded. Inputting as text.")


        if 'Brucella vaccination status (Yes No)' in le_dict:
             vaccination_options = list(le_dict['Brucella vaccination status (Yes No)'].classes_)
             vaccination = st.selectbox("Brucella Vaccination Status", vaccination_options)
        else:
             vaccination = st.text_input("Brucella Vaccination Status")
             st.warning("Brucella vaccination status options could not be loaded. Inputting as text.")


        if 'Sample Type(Serum Milk)' in le_dict:
             sample_options = list(le_dict['Sample Type(Serum Milk)'].classes_)
             sample_type = st.selectbox("Sample Type", sample_options)
        else:
             sample_type = st.text_input("Sample Type")
             st.warning("Sample Type options could not be loaded. Inputting as text.")


    with col3:
        if 'Test Type (RBPT ELISA MRT)' in le_dict:
             test_options = list(le_dict['Test Type (RBPT ELISA MRT)'].classes_)
             test_type = st.selectbox("Test Type", test_options)
        else:
             test_type = st.text_input("Test Type")
             st.warning("Test Type options could not be loaded. Inputting as text.")


        if 'Retained Placenta Stillbirth(Yes No No Data)' in le_dict:
             retained_options = list(le_dict['Retained Placenta Stillbirth(Yes No No Data)'].classes_)
             retained_placenta = st.selectbox("Retained Placenta/Stillbirth", retained_options)
        else:
             retained_placenta = st.text_input("Retained Placenta/Stillbirth")
             st.warning("Retained Placenta/Stillbirth options could not be loaded. Inputting as text.")


        if 'Proper Disposal of Aborted Fetuses (Yes No)' in le_dict:
             disposal_options = list(le_dict['Proper Disposal of Aborted Fetuses (Yes No)'].classes_)
             disposal = st.selectbox("Proper Disposal of Aborted Fetuses", disposal_options)
        else:
             disposal = st.text_input("Proper Disposal of Aborted Fetuses")
             st.warning("Proper Disposal of Aborted Fetuses options could not be loaded. Inputting as text.")


    # Submit button
    submitted = st.form_submit_button("🔍 Predict Brucellosis Status", use_container_width=True)

if submitted:
    try:
        # Prepare input data
        input_data = {
            'Age ': age,
            'Breed species': breed,
            ' Sex ': sex,
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

        # Encode categorical features
        for col in input_df.columns:
            if col in le_dict and input_df[col].dtype == 'object':
                try:
                    # Convert to title case for encoding (assuming this was done during training)
                    input_df[col] = input_df[col].astype(str).str.strip().str.title()
                    input_df[col] = le_dict[col].transform(input_df[col])
                except ValueError:
                    # Handle unseen labels - use a default value (e.g., the most frequent label index or a dedicated 'unseen' category if handled during training)
                    # For simplicity here, we'll just use 0, but a more robust approach might be needed depending on how unseen labels were handled during training.
                    st.warning(f"Unseen label for column '{col}': {input_df[col].values[0]}. Using default encoding 0.")
                    input_df[col] = 0
            elif col in le_dict and not pd.api.types.is_numeric_dtype(input_df[col]):
                 # Handle cases where the column was categorical but input is not object (e.g., numeric but needs encoding)
                 try:
                     input_df[col] = le_dict[col].transform(input_df[col].astype(str).str.strip().str.title())
                 except ValueError:
                     st.warning(f"Unseen or incompatible value for column '{col}': {input_df[col].values[0]}. Using default encoding 0.")
                     input_df[col] = 0


        # Ensure all features are present and in the correct order
        # Add missing columns with a default value (e.g., 0)
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training data order
        input_df = input_df[feature_names]

        # Scale numerical features if a scaler was loaded
        if scaler is not None:
             # Identify numerical columns that were likely scaled during training
             # Assuming 'Age' and 'Calvings' are the numerical columns that were scaled.
             # You should confirm which columns were actually scaled in your training pipeline.
             numerical_cols_to_scale = [col for col in feature_names if col in ['Age ', 'Calvings']]
             input_df[numerical_cols_to_scale] = scaler.transform(input_df[numerical_cols_to_scale])


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
        result_class = "positive-result" if predicted_result == "Positive" else "negative-result" if predicted_result == "Negative" else "suspect-result"

        st.markdown(f"""
        <div class="prediction-box {result_class}">
            <h2>🎯 Predicted Result: {predicted_result}</h2>
            <h3>Confidence: {confidence:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Detailed probabilities
        col1, col2, col3 = st.columns(3)

        # Ensure probabilities are displayed in the correct order of classes
        class_prob_dict = dict(zip(le_target.classes_, probabilities))
        # Sort classes based on the order in le_target.classes_ for consistent display
        sorted_classes = le_target.classes_

        for i, cls in enumerate(sorted_classes):
            prob = class_prob_dict.get(cls, 0.0) # Use .get to handle potential missing classes
            with [col1, col2, col3][i % 3]: # Use modulo 3 to cycle through columns
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

        # Feature importance (loaded from artifacts or general info)
        st.markdown("### 📈 Key Risk Factors")
        # Note: Feature importances from the trained model would be more accurate here.
        # For simplicity, using general indicators.
        # Attempt to load feature importances if model supports it and they were saved
        feature_importance_available = False
        try:
            if model and hasattr(model, 'feature_importances_') and feature_names:
                 feature_importance_scores = model.feature_importances_
                 importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_scores})
                 importance_df = importance_df.sort_values('Importance', ascending=False).head(10)

                 st.write("Top 10 Most Important Features (Based on Trained Model):")
                 st.dataframe(importance_df)
                 feature_importance_available = True
        except:
             pass # Ignore errors if loading/processing feature importances fails


        if not feature_importance_available:
             st.info("""
             Based on general veterinary knowledge and typical Brucellosis risk factors, important factors include:
             - **Abortion History** - Strong indicator
             - **Age** - Older animals at higher risk
             - **Vaccination Status** - Unvaccinated animals at higher risk
             - **Test Type** - ELISA vs RBPT vs MRT
             - **Sample Type** - Serum vs Milk testing
             """)


    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
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


# Sample data upload feature
st.markdown("### 📁 Batch Prediction (Optional)")
uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type="csv")

if uploaded_file is not None:
    try:
        # Read uploaded file
        df_batch = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df_batch.head())

        if st.button("Run Batch Predictions"):
            # Process batch predictions
            predictions = []
            for index, row in df_batch.iterrows():
                try:
                    # Prepare input
                    input_df = pd.DataFrame([row.to_dict()])

                    # Encode categorical features
                    for col in input_df.columns:
                         if col in le_dict and input_df[col].dtype == 'object':
                             try:
                                 # Convert to title case for encoding
                                 input_df[col] = input_df[col].astype(str).str.strip().str.title()
                                 input_df[col] = le_dict[col].transform(input_df[col])
                             except ValueError:
                                 # Handle unseen labels
                                 st.warning(f"Unseen label in batch row {index} for column '{col}': {input_df[col].values[0]}. Using default encoding 0.")
                                 input_df[col] = 0
                         elif col in le_dict and not pd.api.types.is_numeric_dtype(input_df[col]):
                              # Handle cases where the column was categorical but input is not object
                              try:
                                  input_df[col] = le_dict[col].transform(input_df[col].astype(str).str.strip().str.title())
                              except ValueError:
                                  st.warning(f"Unseen or incompatible value in batch row {index} for column '{col}': {input_df[col].values[0]}. Using default encoding 0.")
                                  input_df[col] = 0


                    # Ensure all features are present and in the correct order
                    for col in feature_names:
                        if col not in input_df.columns:
                            input_df[col] = 0

                    input_df = input_df[feature_names]

                    # Scale if needed
                    if scaler is not None:
                         numerical_cols_to_scale = [col for col in feature_names if col in ['Age ', 'Calvings']]
                         input_df[numerical_cols_to_scale] = scaler.transform(input_df[numerical_cols_to_scale])


                    # Predict
                    pred = model.predict(input_df)[0]
                    # Get probability of the predicted class
                    pred_class_index = model.classes_.tolist().index(pred)
                    prob = model.predict_proba(input_df)[0][pred_class_index]


                    predictions.append({
                        'Prediction': le_target.inverse_transform([pred])[0],
                        'Confidence': prob
                    })
                except Exception as e:
                     st.error(f"Error processing row {index}: {row.to_dict()} - {str(e)}")
                     predictions.append({
                         'Prediction': 'Error',
                         'Confidence': 0.0
                     })


            # Display results
            # Ensure original batch data and predictions align correctly
            if len(df_batch) == len(predictions):
                 results_df = pd.concat([df_batch.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
                 st.write("Batch prediction results:")
                 st.dataframe(results_df)

                 # Download results
                 csv = results_df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download Results as CSV",
                     data=csv,
                     file_name="brucellosis_predictions.csv",
                     mime="text/csv"
                 )
            else:
                 st.error("Batch prediction failed for some rows. Results may be incomplete.")
                 st.write("Predictions (partial):")
                 st.dataframe(pd.DataFrame(predictions))


    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
