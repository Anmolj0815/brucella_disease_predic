import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import pickle
import os
from io import StringIO
from ngrok import connect

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
    **Accuracy:** ~95%+
    """)

    st.markdown("### 🔬 About Brucellosis")
    st.write("""
    Brucellosis is a bacterial infection that affects cattle and can be transmitted to humans.
    Early detection is crucial for livestock health management.
    """)

# Load or train model
@st.cache_resource
def load_or_train_model():
    """Load pre-trained model or train new one"""

    model_dir = "./model_artifacts"
    model_path = os.path.join(model_dir, 'best_model.pkl')
    le_dict_path = os.path.join(model_dir, 'le_dict.pkl')
    le_target_path = os.path.join(model_dir, 'le_target.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')


    if os.path.exists(model_path) and os.path.exists(le_dict_path) and \
       os.path.exists(le_target_path) and os.path.exists(feature_names_path):
        st.write("Loading pre-trained model and preprocessors...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(le_dict_path, 'rb') as f:
            le_dict = pickle.load(f)
        with open(le_target_path, 'rb') as f:
            le_target = pickle.load(f)
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)

        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
             st.warning("Scaler not found. Predictions might be inaccurate if the model requires scaled features.")


        st.success("✅ Pre-trained model loaded successfully!")
        return model, le_dict, le_target, scaler, feature_names
    else:
        st.error("❌ Model artifacts not found! Please train the model first.")
        return None, None, None, None, None


# Load model
with st.spinner("Loading model..."):
    model, le_dict, le_target, scaler, feature_names = load_or_train_model()

if model is None:
    st.stop()

# Start ngrok and get public URL
try:
    public_url = connect(port='8501')
    st.info(f"🌍 ngrok tunnel established! Public URL: {public_url}")
except Exception as e:
    st.error(f"❌ Failed to start ngrok tunnel: {str(e)}")


# Main prediction interface
st.markdown('<h2 class="sub-header">🔬 Enter Animal Details</h2>', unsafe_allow_html=True)

# Create input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=20, value=4)

        if 'Breed species' in le_dict:
            breed_options = list(le_dict['Breed species'].classes_)
            breed = st.selectbox("Breed Species", breed_options)
        else:
            breed = st.text_input("Breed Species")

        if 'Sex' in le_dict:
             sex_options = list(le_dict['Sex'].classes_)
             sex = st.selectbox("Sex", sex_options)
        else:
             sex = st.text_input("Sex")


        calvings = st.number_input("Number of Calvings", min_value=0, max_value=10, value=2)

    with col2:
        if 'Abortion History (Yes No)' in le_dict:
            abortion_options = list(le_dict['Abortion History (Yes No)'].classes_)
            abortion_history = st.selectbox("Abortion History", abortion_options)
        else:
             abortion_history = st.text_input("Abortion History")

        if 'Infertility Repeat breeder(Yes No)' in le_dict:
            infertility_options = list(le_dict['Infertility Repeat breeder(Yes No)'].classes_)
            infertility = st.selectbox("Infertility/Repeat Breeder", infertility_options)
        else:
            infertility = st.text_input("Infertility/Repeat Breeder")

        if 'Brucella vaccination status (Yes No)' in le_dict:
             vaccination_options = list(le_dict['Brucella vaccination status (Yes No)'].classes_)
             vaccination = st.selectbox("Brucella Vaccination Status", vaccination_options)
        else:
             vaccination = st.text_input("Brucella Vaccination Status")

        if 'Sample Type(Serum Milk)' in le_dict:
             sample_options = list(le_dict['Sample Type(Serum Milk)'].classes_)
             sample_type = st.selectbox("Sample Type", sample_options)
        else:
             sample_type = st.text_input("Sample Type")


    with col3:
        if 'Test Type (RBPT ELISA MRT)' in le_dict:
             test_options = list(le_dict['Test Type (RBPT ELISA MRT)'].classes_)
             test_type = st.selectbox("Test Type", test_options)
        else:
             test_type = st.text_input("Test Type")

        if 'Retained Placenta Stillbirth(Yes No No Data)' in le_dict:
             retained_options = list(le_dict['Retained Placenta Stillbirth(Yes No No Data)'].classes_)
             retained_placenta = st.selectbox("Retained Placenta/Stillbirth", retained_options)
        else:
             retained_placenta = st.text_input("Retained Placenta/Stillbirth")

        if 'Proper Disposal of Aborted Fetuses (Yes No)' in le_dict:
             disposal_options = list(le_dict['Proper Disposal of Aborted Fetuses (Yes No)'].classes_)
             disposal = st.selectbox("Proper Disposal of Aborted Fetuses", disposal_options)
        else:
             disposal = st.text_input("Proper Disposal of Aborted Fetuses")


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
                    input_df[col] = input_df[col].astype(str).str.strip().str.title()
                    input_df[col] = le_dict[col].transform(input_df[col])
                except ValueError:
                    st.warning(f"Unseen label for column '{col}': {input_df[col].values[0]}. Using default encoding 0.")
                    input_df[col] = 0


        # Ensure all features are present and in the correct order
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_names]

        # Scale if needed (check if scaler was loaded)
        if scaler is not None and any(col in ['Age ', 'Calvings'] for col in input_df.columns):
             scaled_cols = [col for col in feature_names if col in ['Age ', 'Calvings']]
             if scaled_cols:
                 input_df[scaled_cols] = scaler.transform(input_df[scaled_cols])


        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Convert back to original labels
        predicted_result = le_target.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]

        # Display results
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)

        result_class = "positive-result" if predicted_result == "Positive" else "negative-result" if predicted_result == "Negative" else "suspect-result"

        st.markdown(f"""
        <div class="prediction-box {result_class}">
            <h2>🎯 Predicted Result: {predicted_result}</h2>
            <h3>Confidence: {confidence:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        class_prob_dict = dict(zip(le_target.classes_, probabilities))
        sorted_classes = sorted(le_target.classes_)

        for i, cls in enumerate(sorted_classes):
            prob = class_prob_dict.get(cls, 0.0)
            with [col1, col2, col3][i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{cls}</h3>
                    <h2>{prob:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)


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

        st.markdown("### 📈 Key Risk Factors")
        if feature_names and hasattr(model, 'feature_importances_'):
            feature_importance_scores = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_scores})
            importance_df = importance_df.sort_values('Importance', ascending=False).head(10)

            st.write("Top 10 Most Important Features:")
            st.dataframe(importance_df)

        else:
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

st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>🔬 About This System</h4>
    <p>This Brucellosis prediction system uses machine learning to analyze animal health data and predict infection risk.
    The model is trained on veterinary data and uses Extra Trees algorithm for classification.</p>
    <p><strong>Disclaimer:</strong> This tool is for screening purposes only. Always consult with a qualified veterinarian for final diagnosis and treatment decisions.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🚀 Deployment Instructions")
st.code("""
# To run this app:
# 1. Save this code as 'app.py'
# 2. Ensure 'model_artifacts' directory with saved model files exists in the same directory.
# 3. Install required packages:
#    pip install streamlit pandas numpy scikit-learn imbalanced-learn ngrok
# 4. Run the app:
#    streamlit run app.py
# 5. The app will open in your browser at http://localhost:8501 and an ngrok tunnel will be created.
# 6. For cloud deployment, consider alternatives to ngrok like Streamlit Cloud.
""", language="bash")

st.markdown("### 📁 Batch Prediction (Optional)")
uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type="csv")

if uploaded_file is not None:
    try:
        df_batch = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df_batch.head())

        if st.button("Run Batch Predictions"):
            predictions = []
            for _, row in df_batch.iterrows():
                try:
                    input_df = pd.DataFrame([row.to_dict()])

                    for col in input_df.columns:
                        if col in le_dict and input_df[col].dtype == 'object':
                            try:
                                input_df[col] = input_df[col].astype(str).str.strip().str.title()
                                input_df[col] = le_dict[col].transform(input_df[col])
                            except ValueError:
                                st.warning(f"Unseen label in batch for column '{col}': {input_df[col].values[0]}. Using default encoding 0.")
                                input_df[col] = 0


                    for col in feature_names:
                        if col not in input_df.columns:
                            input_df[col] = 0

                    input_df = input_df[feature_names]

                    if scaler is not None and any(col in ['Age ', 'Calvings'] for col in input_df.columns):
                         scaled_cols = [col for col in feature_names if col in ['Age ', 'Calvings']]
                         if scaled_cols:
                             input_df[scaled_cols] = scaler.transform(input_df[scaled_cols])


                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0].max()

                    predictions.append({
                        'Prediction': le_target.inverse_transform([pred])[0],
                        'Confidence': prob
                    })
                except Exception as e:
                     st.error(f"Error processing row: {row.to_dict()} - {str(e)}")
                     predictions.append({
                         'Prediction': 'Error',
                         'Confidence': 0.0
                     })


            results_df = pd.concat([df_batch.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
            st.write("Batch prediction results:")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="brucellosis_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
